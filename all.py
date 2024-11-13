import torch
from ultralytics import YOLO
import xml.etree.ElementTree as ET
import numpy as np
import os, shutil, cv2
# from os import listdir, getcwd
# from os.path import join
# import glob
import yaml
import random


def get_classnames(yaml_file):
    name_list = []
    file = open(yaml_file, 'r', encoding="utf-8")
    # file_data = file.read()
    file_data = yaml.load(file, yaml.FullLoader)
    file.close()
    names_dict = file_data["names"]
    for k,v in names_dict.items():
        print(v)
        name_list.append(v)
    return name_list

def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(yaml_file, path, savepath):
    classes = get_classnames(yaml_file)
    filenames = os.listdir(path)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for image_name in filenames:
        # print(image_name)
        in_file = open(os.path.join(path, image_name), 'r', encoding='utf-8')
        xml_text = in_file.read()
        root = ET.fromstring(xml_text)
        in_file.close()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        out_file = open(os.path.join(savepath, image_name[:-4] + '.txt'), 'w', encoding='utf-8')
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                print('Not exist in Classes  ' ,image_name, cls)
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        out_file.close()

def check_image_txt_pair( file_path, txt_path ):
    all_images,  all_labels = [], []
    for label in os.listdir(txt_path):
        label_path = os.path.join(txt_path, label)
        in_file = open(label_path, 'r', encoding='utf-8')
        lines = in_file.readlines()
        in_file.close()
        if len(lines):
            image = label.replace('.txt', '.jpg')
            image_path = os.path.join(file_path, image)
            im = cv2.imread(image_path)
            if im is not None:
                all_images.append(image)
                all_labels.append(label)
    return all_images,  all_labels


def split_data(file_path, txt_path, new_file_path, types, ratios):
    all_images, all_labels = check_image_txt_pair(file_path,txt_path)
    assert len(all_images) == len(all_labels)
    total = len(all_images)
    print('total image-txt-pair num : ', total)

    data = list(zip(all_images, all_labels))
    random.shuffle(data)
    each_class_image, each_class_label = zip(*data)

    index_list = [0]
    cnt = 0
    for r in range(len(ratios)-1):
        cnt += int(total * ratios[r])
        index_list.append(cnt)
    index_list.append(total)
    print(index_list)
    for i in range(len(types)):
        start_index, end_index = index_list[i], index_list[i+1]
        images = each_class_image[start_index:end_index]
        labels = each_class_label[start_index:end_index]
        new_imgfoldpath = os.path.join(new_file_path, 'images',types[i])
        if not os.path.exists(new_imgfoldpath):
            os.makedirs(new_imgfoldpath)
        new_txtfoldpath = os.path.join(new_file_path, 'labels', types[i])
        if not os.path.exists(new_txtfoldpath):
            os.makedirs(new_txtfoldpath)
        c = 0
        for image,label in zip(images, labels):
            old_imgpath = os.path.join(file_path, image)
            old_txtpath = os.path.join(txt_path, label)
            new_imgpath = os.path.join(new_imgfoldpath, image)
            new_txtpath = os.path.join(new_txtfoldpath, label)
            shutil.copy(old_imgpath, new_imgpath)
            shutil.copy(old_txtpath, new_txtpath)
            c += 1
        print(types[i] + ' image-txt-pair num : ', c)



def train(model, data_root, batch_size, img_size, max_epoch, project_name, save_weight):
    results = model.train(
        data=data_root,
        epochs=max_epoch,
        imgsz=img_size,  # imgsz应该尽量贴近训练集图片的大小，但是要是32的倍数
        plots=True,
        batch=batch_size,
        amp=False,
        # fraction=0.1, # 设置fraction参数用于只训练数据集的一部分，设置0.1表示只训练10%的数据集
        project= project_name,
        patience=30,
        # degrees=180,
        auto_augment="autoaugment",
        cache=True,  # 缓存数据集，加快训练速度
        hsv_h=0.02,  # 调整图像的色调，对提高模型对不同颜色的物体的识别能力有帮助
        translate=0.2,  # 平移图像，帮助模型识别边角的物体
        flipud=0.5,  # 以指定概率上下翻转图像
        # bgr=0.5, # 以指定概率随机改变图像的颜色通道顺序，提高模型对不同颜色的物体的识别能力
        close_mosaic=20,  # 最后稳定训练
        scale=0.3,
        device=0,
        workers=0,
        name=save_weight,
        resume=False,  # 断点训练，默认Flase

    )

def evaluation(model, data_root, batch_size, img_size):
    # 加载模型，split='test'利用测试集进行测试
    model.val(data=data_root,
              # split='test',
              imgsz=img_size,
              batch=batch_size,
              device=0,
              workers=0,
              save=True,
              task="test")

def over_message():
    width, height = 640, 480
    blank_image = np.zeros((height, width, 3), np.uint8)

    # 设置文字"over"的位置，这里设置在图片的正中央
    text = "over"
    position = (width // 2 - 20, height // 2 + 10)

    # 设置文字的字体、大小、颜色和背景颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    background_color = (0, 0, 0)  # 黑色

    # 在空白图片上添加文字
    cv2.putText(blank_image, text, position, font, font_scale, font_color, 2)

    # 显示图片
    # cv2.imshow("Blank Image", blank_image)
    # cv2.waitKey(0)

    # 保存图片
    cv2.imwrite("Over.jpg", blank_image)







if __name__ == '__main__':

    abs_path = "E:\\Work\\AIADC\\Trains_V1\\Recipe"
    data_yaml_file = os.path.join(abs_path, "Data", "SN01.yaml")
    imgs_data = os.path.join(abs_path, "Data", "samples")
    xmls_data = os.path.join(abs_path,"Data",  "xmls")
    txts_data = os.path.join(abs_path, "Data", "temp_txts")
    save_path = os.path.join(abs_path, "Data", "SN01")

    dataset_types = ['train', 'val', 'test']
    dataset_ratio = [0.8, 0.1, 0.1]

    if os.path.exists(txts_data):
        shutil.rmtree(txts_data)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    convert_annotation(data_yaml_file, xmls_data, txts_data)
    split_data(imgs_data, txts_data, save_path, dataset_types, dataset_ratio)

#     Processed Data Done    ###########

    init_weight = os.path.join(abs_path, "pretrained", "yolo11n.pt")
    model = YOLO(init_weight)
    data_path = os.path.join(abs_path, "Data",  "SN01.yaml" )
    batch_size = 4
    project_name = "runs"
    save_log = "exp"
    img_size = 640
    max_epoch = 3
    backup = "backup"
    temp_modeldir = os.path.join(abs_path, project_name)
    backup_dir = os.path.join(abs_path, backup)
    if os.path.exists(temp_modeldir):
        shutil.rmtree(temp_modeldir)
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    train(model, data_path, batch_size, img_size, max_epoch, project_name, save_log)

    best_weight = os.path.join(temp_modeldir, "exp", "weights", "best.pt")
    best_model = YOLO(best_weight)
    best_model.export(format='onnx', opset=11)

    evaluation(best_weight, data_path, batch_size, img_size)
    shutil.copytree(temp_modeldir, backup_dir)
    over_message()








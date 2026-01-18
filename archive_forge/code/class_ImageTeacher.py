from parlai.core.teachers import (
from parlai.core.opt import Opt
import copy
import random
import itertools
import os
from PIL import Image
import string
import json
from abc import ABC
from typing import Tuple, List
class ImageTeacher(AbstractImageTeacher):
    """
    Teacher which provides images and captions.

    In __init__, setup some fake images + features
    """

    def __init__(self, opt, shared=None):
        self._setup_test_data(opt)
        super().__init__(opt, shared)

    def _setup_test_data(self, opt):
        datapath = os.path.join(opt['datapath'], 'ImageTeacher')
        imagepath = os.path.join(datapath, 'images')
        os.makedirs(imagepath, exist_ok=True)
        self.image_features_path = os.path.join(datapath, f'{opt['image_mode']}_image_features')
        imgs = [f'img_{i}' for i in range(10)]
        for i, img in enumerate(imgs):
            image = Image.new('RGB', (16, 16), color=i)
            image.save(os.path.join(imagepath, f'{img}.jpg'), 'JPEG')
        for dt in ['train', 'valid', 'test']:
            random.seed(42)
            data = [{'image_id': img, 'text': string.ascii_uppercase[i]} for i, img in enumerate(imgs)]
            with open(os.path.join(datapath, f'{dt}.json'), 'w') as f:
                json.dump(data, f)

    def get_image_features_path(self, task, image_model_name, dt):
        """
        Return path dummy image features.
        """
        return self.image_features_path

    def image_id_to_image_path(self, image_id):
        """
        Return path to image on disk.
        """
        return os.path.join(self.opt['datapath'], 'ImageTeacher/images', f'{image_id}.jpg')
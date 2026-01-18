import parlai.core.build_data as build_data
import parlai.utils.logging as logging
import os
from PIL import Image
from zipfile import ZipFile
def _init_transform(self):
    self.transform = self.transforms.Compose([self.transforms.Scale(self.image_size), self.transforms.CenterCrop(self.crop_size), self.transforms.ToTensor(), self.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
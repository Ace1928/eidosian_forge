import codecs
import os
import os.path
import shutil
import string
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError
import numpy as np
import torch
from PIL import Image
from .utils import _flip_byte_order, check_integrity, download_and_extract_archive, extract_archive, verify_str_arg
from .vision import VisionDataset
def _load_data(self):
    data = read_sn3_pascalvincent_tensor(self.images_file)
    if data.dtype != torch.uint8:
        raise TypeError(f'data should be of dtype torch.uint8 instead of {data.dtype}')
    if data.ndimension() != 3:
        raise ValueError('data should have 3 dimensions instead of {data.ndimension()}')
    targets = read_sn3_pascalvincent_tensor(self.labels_file).long()
    if targets.ndimension() != 2:
        raise ValueError(f'targets should have 2 dimensions instead of {targets.ndimension()}')
    if self.what == 'test10k':
        data = data[0:10000, :, :].clone()
        targets = targets[0:10000, :].clone()
    elif self.what == 'test50k':
        data = data[10000:, :, :].clone()
        targets = targets[10000:, :].clone()
    return (data, targets)
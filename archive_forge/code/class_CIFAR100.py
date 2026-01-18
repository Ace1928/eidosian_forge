import os.path
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from .utils import check_integrity, download_and_extract_archive
from .vision import VisionDataset
class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    filename = 'cifar-100-python.tar.gz'
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [['train', '16019d7e3df5f24257cddd939b257f8d']]
    test_list = [['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc']]
    meta = {'filename': 'meta', 'key': 'fine_label_names', 'md5': '7973b15100ade9c7d40fb424638fde48'}
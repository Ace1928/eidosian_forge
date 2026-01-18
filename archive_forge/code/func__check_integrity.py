import csv
import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple, Union
import PIL
import torch
from .utils import check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg
from .vision import VisionDataset
def _check_integrity(self) -> bool:
    for _, md5, filename in self.file_list:
        fpath = os.path.join(self.root, self.base_folder, filename)
        _, ext = os.path.splitext(filename)
        if ext not in ['.zip', '.7z'] and (not check_integrity(fpath, md5)):
            return False
    return os.path.isdir(os.path.join(self.root, self.base_folder, 'img_align_celeba'))
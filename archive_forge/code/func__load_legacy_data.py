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
def _load_legacy_data(self):
    data_file = self.training_file if self.train else self.test_file
    return torch.load(os.path.join(self.processed_folder, data_file))
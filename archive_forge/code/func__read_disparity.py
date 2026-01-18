import functools
import json
import os
import random
import shutil
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Callable, cast, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from .utils import _read_pfm, download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
def _read_disparity(self, file_path: str) -> Union[Tuple[None, None], Tuple[np.ndarray, np.ndarray]]:
    if file_path is None:
        return (None, None)
    disparity_map = _read_pfm_file(file_path)
    disparity_map = np.abs(disparity_map)
    mask_path = Path(file_path).parent / 'mask0nocc.png'
    valid_mask = Image.open(mask_path)
    valid_mask = np.asarray(valid_mask).astype(bool)
    return (disparity_map, valid_mask)
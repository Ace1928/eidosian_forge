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
def _read_img(self, file_path: Union[str, Path]) -> Image.Image:
    """
        Function that reads either the original right image or an augmented view when ``use_ambient_views`` is True.
        When ``use_ambient_views`` is True, the dataset will return at random one of ``[im1.png, im1E.png, im1L.png]``
        as the right image.
        """
    ambient_file_paths: List[Union[str, Path]]
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if file_path.name == 'im1.png' and self.use_ambient_views:
        base_path = file_path.parent
        ambient_file_paths = list((base_path / view_name for view_name in ['im1E.png', 'im1L.png']))
        ambient_file_paths = list(filter(lambda p: os.path.exists(p), ambient_file_paths))
        ambient_file_paths.append(file_path)
        file_path = random.choice(ambient_file_paths)
    return super()._read_img(file_path)
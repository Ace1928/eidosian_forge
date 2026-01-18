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
def _get_occlussion_mask_paths(self, file_path: str) -> Tuple[str, str]:
    fpath = Path(file_path)
    basename = fpath.name
    scenedir = fpath.parent
    sampledir = scenedir.parent.parent
    occlusion_path = str(sampledir / 'occlusions' / scenedir.name / basename)
    outofframe_path = str(sampledir / 'outofframe' / scenedir.name / basename)
    if not os.path.exists(occlusion_path):
        raise FileNotFoundError(f'Occlusion mask {occlusion_path} does not exist')
    if not os.path.exists(outofframe_path):
        raise FileNotFoundError(f'Out of frame mask {outofframe_path} does not exist')
    return (occlusion_path, outofframe_path)
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
def _scan_pairs(self, paths_left_pattern: str, paths_right_pattern: Optional[str]=None) -> List[Tuple[str, Optional[str]]]:
    left_paths = list(sorted(glob(paths_left_pattern)))
    right_paths: List[Union[None, str]]
    if paths_right_pattern:
        right_paths = list(sorted(glob(paths_right_pattern)))
    else:
        right_paths = list((None for _ in left_paths))
    if not left_paths:
        raise FileNotFoundError(f'Could not find any files matching the patterns: {paths_left_pattern}')
    if not right_paths:
        raise FileNotFoundError(f'Could not find any files matching the patterns: {paths_right_pattern}')
    if len(left_paths) != len(right_paths):
        raise ValueError(f'Found {len(left_paths)} left files but {len(right_paths)} right files using:\n left pattern: {paths_left_pattern}\nright pattern: {paths_right_pattern}\n')
    paths = list(((left, right) for left, right in zip(left_paths, right_paths)))
    return paths
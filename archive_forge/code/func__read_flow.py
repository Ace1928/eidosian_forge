import itertools
import os
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from ..io.image import _read_png_16
from .utils import _read_pfm, verify_str_arg
from .vision import VisionDataset
def _read_flow(self, file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    return _read_16bits_png_with_flow_and_valid_mask(file_name)
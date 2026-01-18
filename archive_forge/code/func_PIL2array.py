import os
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from .utils import download_url
from .vision import VisionDataset
def PIL2array(_img: Image.Image) -> np.ndarray:
    """Convert PIL image type to numpy 2D array"""
    return np.array(_img.getdata(), dtype=np.uint8).reshape(64, 64)
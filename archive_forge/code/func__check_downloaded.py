import os
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from .utils import download_url
from .vision import VisionDataset
def _check_downloaded(self) -> bool:
    return os.path.exists(self.data_dir)
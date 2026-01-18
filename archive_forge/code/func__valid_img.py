import inspect
import os
from typing import List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
def _valid_img(img: Tensor, normalize: bool) -> bool:
    """Check that input is a valid image to the network."""
    value_check = img.max() <= 1.0 and img.min() >= 0.0 if normalize else img.min() >= -1
    return img.ndim == 4 and img.shape[1] == 3 and value_check
import inspect
import os
from typing import List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
def _spatial_average(in_tens: Tensor, keep_dim: bool=True) -> Tensor:
    """Spatial averaging over height and width of images."""
    return in_tens.mean([2, 3], keepdim=keep_dim)
import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, grid_sample, interpolate, pad as torch_pad
def _assert_channels(img: Tensor, permitted: List[int]) -> None:
    c = get_dimensions(img)[0]
    if c not in permitted:
        raise TypeError(f'Input image tensor permitted channel values are {permitted}, but found {c}')
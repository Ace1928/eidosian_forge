import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, grid_sample, interpolate, pad as torch_pad
def _equalize_single_image(img: Tensor) -> Tensor:
    return torch.stack([_scale_channel(img[c]) for c in range(img.size(0))])
import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
from .nested_tensor import NestedTensor
def _pad_last_dim(tensor: torch.Tensor, alignment_size: int, slice: bool) -> torch.Tensor:
    last_dim_size = tensor.size(-1)
    if last_dim_size % alignment_size == 0:
        return tensor
    pad_count = alignment_size - last_dim_size % alignment_size
    tensor = torch.nn.functional.pad(tensor, [0, pad_count])
    if slice:
        return tensor[..., 0:last_dim_size]
    return tensor
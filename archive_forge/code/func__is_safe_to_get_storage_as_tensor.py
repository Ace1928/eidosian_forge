import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
from .nested_tensor import NestedTensor
def _is_safe_to_get_storage_as_tensor(tensor: torch.Tensor):
    assert isinstance(tensor, NestedTensor)
    offsets = tensor.offsets()
    strides = tensor._stride
    n_tensors = offsets.size(0) - 1
    if n_tensors <= 1:
        return True
    prev_stride = strides[1]
    for stride in strides[2:]:
        if prev_stride <= stride:
            return False
        prev_stride = stride
    return True
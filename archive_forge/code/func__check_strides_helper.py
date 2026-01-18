from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def _check_strides_helper(a: TensorLikeType, b: TensorLikeType, *, only_cuda=True, significant_only=True) -> Tuple[bool, Optional[int]]:
    if (not only_cuda or a.device.type == 'cuda' or b.device.type == 'cuda') and a.numel() > 0:
        for idx in range(a.ndim):
            check = not significant_only or a.shape[idx] > 1
            if a.stride()[idx] != b.stride()[idx] and check:
                return (False, idx)
    return (True, None)
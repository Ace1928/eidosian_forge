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
def _find_highest_dtype_filtered(args, filter, *, float_as_complex=False) -> Optional[torch.dtype]:
    zero_dim_tensor_dtype = None
    one_plus_dim_tensor_dtype = None
    for x in args:
        if isinstance(x, TensorLike) and filter(x.dtype):
            _dtype = x.dtype
            if float_as_complex and is_float_dtype(_dtype):
                _dtype = corresponding_complex_dtype(_dtype)
            if x.ndim == 0:
                zero_dim_tensor_dtype = get_higher_dtype(zero_dim_tensor_dtype, _dtype)
            else:
                one_plus_dim_tensor_dtype = get_higher_dtype(one_plus_dim_tensor_dtype, _dtype)
    if one_plus_dim_tensor_dtype is not None:
        return one_plus_dim_tensor_dtype
    return zero_dim_tensor_dtype
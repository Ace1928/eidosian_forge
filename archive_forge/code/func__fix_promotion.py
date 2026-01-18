from __future__ import annotations
from functools import wraps
from builtins import all as builtin_all, any as builtin_any
from ..common._aliases import (UniqueAllResult, UniqueCountsResult,
from .._internal import get_xp
import torch
from typing import TYPE_CHECKING
def _fix_promotion(x1, x2, only_scalar=True):
    if x1.dtype not in _array_api_dtypes or x2.dtype not in _array_api_dtypes:
        return (x1, x2)
    if not only_scalar or x1.shape == ():
        dtype = result_type(x1, x2)
        x2 = x2.to(dtype)
    if not only_scalar or x2.shape == ():
        dtype = result_type(x1, x2)
        x1 = x1.to(dtype)
    return (x1, x2)
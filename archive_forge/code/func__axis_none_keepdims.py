from __future__ import annotations
from functools import wraps
from builtins import all as builtin_all, any as builtin_any
from ..common._aliases import (UniqueAllResult, UniqueCountsResult,
from .._internal import get_xp
import torch
from typing import TYPE_CHECKING
def _axis_none_keepdims(x, ndim, keepdims):
    if keepdims:
        for i in range(ndim):
            x = torch.unsqueeze(x, 0)
    return x
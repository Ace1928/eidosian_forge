from __future__ import annotations
import builtins
import itertools
import operator
from typing import Optional, Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import (
from torch import broadcast_shapes
def _tensor_equal(a1, a2, equal_nan=False):
    if a1.shape != a2.shape:
        return False
    cond = a1 == a2
    if equal_nan:
        cond = cond | torch.isnan(a1) & torch.isnan(a2)
    return cond.all().item()
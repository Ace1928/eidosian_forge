from __future__ import annotations
import builtins
import itertools
import operator
from typing import Optional, Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import (
from torch import broadcast_shapes
def _split_helper_int(tensor, indices_or_sections, axis, strict=False):
    if not isinstance(indices_or_sections, int):
        raise NotImplementedError('split: indices_or_sections')
    axis = _util.normalize_axis_index(axis, tensor.ndim)
    l, n = (tensor.shape[axis], indices_or_sections)
    if n <= 0:
        raise ValueError()
    if l % n == 0:
        num, sz = (n, l // n)
        lst = [sz] * num
    else:
        if strict:
            raise ValueError('array split does not result in an equal division')
        num, sz = (l % n, l // n + 1)
        lst = [sz] * num
    lst += [sz - 1] * (n - num)
    return torch.split(tensor, lst, axis)
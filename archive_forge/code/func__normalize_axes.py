from __future__ import annotations
from functools import wraps
from builtins import all as builtin_all, any as builtin_any
from ..common._aliases import (UniqueAllResult, UniqueCountsResult,
from .._internal import get_xp
import torch
from typing import TYPE_CHECKING
def _normalize_axes(axis, ndim):
    axes = []
    if ndim == 0 and axis:
        raise IndexError(f'Dimension out of range: {axis[0]}')
    lower, upper = (-ndim, ndim - 1)
    for a in axis:
        if a < lower or a > upper:
            raise IndexError(f'Dimension out of range (expected to be in range of [{lower}, {upper}], but got {a}')
        if a < 0:
            a = a + ndim
        if a in axes:
            raise IndexError(f'Axis {a} appears multiple times in the list of axes')
        axes.append(a)
    return sorted(axes)
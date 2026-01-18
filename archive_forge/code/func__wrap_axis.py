from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
def _wrap_axis(axis, ndim):
    if not -ndim <= axis < ndim:
        raise ValueError(f'invalid axis {axis}. Expected {-ndim} <= axis < {ndim}')
    return axis if axis >= 0 else axis + ndim
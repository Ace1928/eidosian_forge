from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
def _shape_check_impl(shape):
    shape = _constexpr_to_value(shape)
    for i, d in enumerate(shape):
        if isinstance(d, int):
            d = constexpr(d)
        if not isinstance(d, constexpr):
            raise TypeError(f'Shape element {i} must have type `constexpr`')
        if not isinstance(d.value, int):
            raise TypeError(f'Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]')
    return [_constexpr_to_value(x) for x in shape]
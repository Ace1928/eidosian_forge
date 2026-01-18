from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def _promote_reduction_input(t, _builder=None):
    scalar_ty = t.type.scalar
    if scalar_ty is bfloat16:
        return t.to(float32, _builder=_builder)
    return t
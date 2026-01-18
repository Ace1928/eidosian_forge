from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@contextmanager
def _insertion_guard(builder):
    ip = builder.get_insertion_point()
    yield
    builder.restore_insertion_point(ip)
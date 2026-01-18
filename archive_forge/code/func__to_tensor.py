from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
def _to_tensor(x, builder):
    if isinstance(x, bool):
        return tensor(builder.get_int1(x), int1)
    elif isinstance(x, int):
        if -2 ** 31 <= x < 2 ** 31:
            return tensor(builder.get_int32(x), int32)
        elif 2 ** 31 <= x < 2 ** 32:
            return tensor(builder.get_uint32(x), uint32)
        elif -2 ** 63 <= x < 2 ** 63:
            return tensor(builder.get_int64(x), int64)
        elif 2 ** 63 <= x < 2 ** 64:
            return tensor(builder.get_uint64(x), uint64)
        else:
            raise RuntimeError(f'Nonrepresentable integer {x}.')
    elif isinstance(x, float):
        min_float32 = 2 ** (-126)
        max_float32 = (2 - 2 ** (-23)) * 2 ** 127
        abs_x = __builtins__['abs'](x)
        if abs_x == float('inf') or abs_x == 0.0 or x != x or (min_float32 <= abs_x <= max_float32):
            return tensor(builder.get_fp32(x), float32)
        else:
            return tensor(builder.get_fp64(x), float64)
    elif isinstance(x, constexpr):
        return _to_tensor(x.value, builder)
    elif isinstance(x, tensor):
        return x
    assert False, f'cannot convert {x} of type {type(x)} to tensor'
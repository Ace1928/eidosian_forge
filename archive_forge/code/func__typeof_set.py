from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(set)
def _typeof_set(val, c):
    if len(val) == 0:
        raise ValueError('Cannot type empty set')
    item = next(iter(val))
    ty = typeof_impl(item, c)
    if ty is None:
        raise ValueError(f'Cannot type set element type {type(item)}')
    return types.Set(ty, reflected=True)
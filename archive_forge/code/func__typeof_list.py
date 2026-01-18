from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(list)
def _typeof_list(val, c):
    if len(val) == 0:
        raise ValueError('Cannot type empty list')
    ty = typeof_impl(val[0], c)
    if ty is None:
        raise ValueError(f'Cannot type list element type {type(val[0])}')
    return types.List(ty, reflected=True)
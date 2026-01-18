from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(tuple)
def _typeof_tuple(val, c):
    tys = [typeof_impl(v, c) for v in val]
    if any((ty is None for ty in tys)):
        return
    return types.BaseTuple.from_types(tys, type(val))
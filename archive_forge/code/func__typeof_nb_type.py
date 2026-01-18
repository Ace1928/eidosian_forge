from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(types.Type)
def _typeof_nb_type(val, c):
    if isinstance(val, types.BaseFunction):
        return val
    elif isinstance(val, (types.Number, types.Boolean)):
        return types.NumberClass(val)
    else:
        return types.TypeRef(val)
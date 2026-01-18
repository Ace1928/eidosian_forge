from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(ctypes._CFuncPtr)
def _typeof_ctypes_function(val, c):
    from .ctypes_utils import is_ctypes_funcptr, make_function_type
    if is_ctypes_funcptr(val):
        return make_function_type(val)
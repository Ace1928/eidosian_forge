from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(slice)
def _typeof_slice(val, c):
    return types.slice2_type if val.step in (None, 1) else types.slice3_type
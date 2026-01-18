from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(np.generic)
def _typeof_numpy_scalar(val, c):
    try:
        return numpy_support.map_arrayscalar_type(val)
    except NotImplementedError:
        pass
from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(np.ndarray)
def _typeof_ndarray(val, c):
    if isinstance(val, np.ma.MaskedArray):
        msg = 'Unsupported array type: numpy.ma.MaskedArray.'
        raise errors.NumbaTypeError(msg)
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except errors.NumbaNotImplementedError:
        raise errors.NumbaValueError(f'Unsupported array dtype: {val.dtype}')
    layout = numpy_support.map_layout(val)
    readonly = not val.flags.writeable
    return types.Array(dtype, val.ndim, layout, readonly=readonly)
from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
def _typeof_buffer(val, c):
    from numba.core.typing import bufproto
    try:
        m = memoryview(val)
    except TypeError:
        return
    try:
        dtype = bufproto.decode_pep3118_format(m.format, m.itemsize)
    except ValueError:
        return
    type_class = bufproto.get_type_class(type(val))
    layout = bufproto.infer_layout(m)
    return type_class(dtype, m.ndim, layout=layout, readonly=m.readonly)
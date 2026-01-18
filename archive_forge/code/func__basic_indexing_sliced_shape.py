from array import array as native_array
import ctypes
import warnings
import operator
from functools import reduce # pylint: disable=redefined-builtin
import numpy as np
from ..base import _LIB, numeric_types, integer_types
from ..base import c_str, c_array, c_array_buf, c_handle_array, mx_real_t
from ..base import mx_uint, NDArrayHandle, check_call, DLPackHandle, mx_int, mx_int64
from ..base import ctypes2buffer
from ..runtime import Features
from ..context import Context, current_context
from ..util import is_np_array
from . import _internal
from . import op
from ._internal import NDArrayBase
@staticmethod
def _basic_indexing_sliced_shape(slc_key, shape):
    """Return the shape after slicing with the given key."""
    assert len(slc_key) == len(shape)
    sliced_shape = []
    for slc, n in zip(slc_key, shape):
        num_elements = _get_slice_len(slc, n)
        sliced_shape.append(num_elements)
    return tuple(sliced_shape)
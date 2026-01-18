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
def _basic_indexing_contiguous_flat_begin_end(slc_key, shape):
    """Return the flat indices of begin and end for contiguous slicing."""
    assert len(slc_key) == len(shape)
    flat_begin, flat_end = (0, 0)
    for slc, n in zip(slc_key, shape):
        flat_begin *= n
        flat_end *= n
        begin, _, _ = slc.indices(n)
        num_elements = _get_slice_len(slc, n)
        if num_elements == 0:
            return (0, 0)
        else:
            flat_begin += begin
            flat_end += begin + num_elements - 1
    return (flat_begin, flat_end + 1)
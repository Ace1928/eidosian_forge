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
def _get_dim_size(start, stop, step):
    """Given start, stop, and step, calculate the number of elements
    of this slice.
    """
    assert step != 0
    if stop == start:
        return 0
    if step > 0:
        assert start < stop
        dim_size = (stop - start - 1) // step + 1
    else:
        assert stop < start
        dim_size = (start - stop - 1) // -step + 1
    return dim_size
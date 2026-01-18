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
def _drop_slice_none_at_end(key):
    """Remove ``slice(None)`` at the end of a key.

        This is used for efficiency in advanced indexing, to avoid generating
        ``arange(n)`` arrays for these axes. The `gather_nd` and `scatter_nd`
        handle implicit full trailing axes automatically.
        """
    key = list(key)
    while isinstance(key[-1], py_slice) and key[-1] == slice(None):
        key.pop()
    return tuple(key)
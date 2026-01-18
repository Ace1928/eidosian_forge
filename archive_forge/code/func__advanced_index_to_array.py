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
def _advanced_index_to_array(idx, ax_len, ctx):
    """Convert ``idx`` to `NDArray` for advanced indexing.

        The ``ax_len`` is used to convert `slice` objects to integer arrays.
        """
    if _int64_enabled():
        idx_dtype = 'int64'
    else:
        idx_dtype = 'int32'
    if isinstance(idx, NDArray):
        if idx.dtype != idx_dtype:
            idx = idx.astype(idx_dtype)
        return idx.as_in_context(ctx)
    elif isinstance(idx, (np.ndarray, list, tuple)):
        return array(idx, ctx, idx_dtype)
    elif isinstance(idx, integer_types):
        return array([idx], ctx, idx_dtype)
    elif isinstance(idx, py_slice):
        start, stop, step = idx.indices(ax_len)
        return arange(start, stop, step, ctx=ctx, dtype=idx_dtype)
    elif isinstance(idx, range):
        return arange(idx.start, idx.stop, idx.step, ctx=ctx, dtype=idx_dtype)
    else:
        raise RuntimeError('illegal index type {}'.format(type(idx)))
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
def _shape_for_bcast(shape, target_ndim, new_axes):
    """Return shape with added axes for broadcasting in ``target_ndim`` dimensions.

    If ``shape`` is shorter than ``target_ndim``, fixed ``1`` entries are inserted
    into the returned shape, in locations indexed by ``new_axes``. The rest is
    filled from the back with ``shape`` while possible.
    """
    new_shape = [None] * target_ndim
    if len(shape) < target_ndim:
        for new_ax in new_axes:
            new_shape[new_ax] = 1
    ax_s = 1
    for ax in range(1, target_ndim + 1):
        if new_shape[-ax] is None:
            try:
                new_shape[-ax] = shape[-ax_s]
                ax_s += 1
            except IndexError:
                new_shape[-ax] = 1
    return tuple(new_shape)
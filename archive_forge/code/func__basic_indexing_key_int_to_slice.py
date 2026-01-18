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
def _basic_indexing_key_int_to_slice(idcs):
    """Return the converted indexing tuple and the integer axes."""
    int_axes = []
    conv_idcs = []
    for ax, idx in enumerate(idcs):
        if isinstance(idx, integer_types):
            conv_idcs.append(_int_to_slice(idx))
            int_axes.append(ax)
        else:
            conv_idcs.append(idx)
    return (tuple(conv_idcs), tuple(int_axes))
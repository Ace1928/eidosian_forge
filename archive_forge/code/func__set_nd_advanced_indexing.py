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
def _set_nd_advanced_indexing(self, key, value):
    """This function is called by __setitem__ when key is an advanced index."""
    indices, new_axes = self._get_index_nd(key)
    vshape = get_oshape_of_gather_nd_op(self.shape, indices.shape)
    value_nd = self._prepare_value_nd(value, bcast_shape=vshape, squeeze_axes=new_axes)
    self._scatter_set_nd(value_nd, indices)
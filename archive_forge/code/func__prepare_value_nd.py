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
def _prepare_value_nd(self, value, bcast_shape, squeeze_axes=None):
    """Return a broadcast `NDArray` with same context and dtype as ``self``.
        For setting item, The returned `ndarray` is squeezed according to squeeze_axes since the
        value_nd is assigned to not yet expanded space in original array.
        `value`: numeric types or array like.
        `bcast_shape`: a shape tuple.
        `squeeze_axes`: a sequence of axes to squeeze in the value array.
        """
    if isinstance(value, numeric_types):
        value_nd = full(bcast_shape, value, ctx=self.ctx, dtype=self.dtype)
    elif type(value) == self.__class__:
        value_nd = value.as_in_context(self.ctx)
        if value_nd.dtype != self.dtype:
            value_nd = value_nd.astype(self.dtype)
    else:
        try:
            value_nd = array(value, ctx=self.ctx, dtype=self.dtype)
        except:
            raise TypeError('{} does not support assignment with non-array-like object {} of type {}'.format(self.__class__, value, type(value)))
    if squeeze_axes and value_nd.ndim > len(bcast_shape):
        squeeze_axes = tuple([ax for ax in squeeze_axes if ax < len(value_nd.shape)])
        value_nd = value_nd.squeeze(axis=tuple(squeeze_axes))
    if value_nd.ndim > len(bcast_shape):
        squeeze_axes = []
        for i in range(value_nd.ndim - len(bcast_shape)):
            if value_nd.shape[i] == 1:
                squeeze_axes.append(i)
            else:
                break
        if squeeze_axes:
            value_nd = value_nd.squeeze(squeeze_axes)
    if value_nd.shape != bcast_shape:
        if value_nd.size == 0:
            value_nd = value_nd.reshape(bcast_shape)
        else:
            value_nd = value_nd.broadcast_to(bcast_shape)
    return value_nd
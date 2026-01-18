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
def _set_nd_basic_indexing(self, key, value):
    """This function indexes ``self`` with a tuple of ``slice`` objects only."""
    for idx in key:
        if idx is not None and (not isinstance(idx, (py_slice, integer_types))):
            raise RuntimeError('`key` may only contain `slice` or integer objects in the basic implementation, got object of type {}. This is a bug, please report it!'.format(type(idx)))
    key_nd = tuple((idx for idx in key if idx is not None))
    int_axes = [ax for ax in range(len(key_nd)) if isinstance(key_nd[ax], integer_types)]
    for ax in int_axes:
        if not -self.shape[ax] <= key_nd[ax] < self.shape[ax]:
            raise IndexError('index {} is out of bounds for axis {} with size {}'.format(key_nd[ax], ax, self.shape[ax]))
    begin, end, step = self._basic_indexing_key_to_begin_end_step(key, self.shape, keep_none=False)
    indexed_shape = tuple((_get_dim_size(b, e, s) for b, e, s in zip(begin, end, step)))
    can_assign_directly = indexed_shape == self.shape and all((s > 0 for s in step))
    begin, end, step = self._basic_indexing_key_to_begin_end_step(key, self.shape, keep_none=True)
    none_axes = [ax for ax in range(len(key)) if key[ax] is None]
    new_axes = self._new_axes_after_basic_indexing(none_axes, key)
    if can_assign_directly:
        if type(value) == self.__class__:
            if value.handle is not self.handle:
                bcast_shape = self._drop_int_axes(indexed_shape, int_axes)
                value_nd = self._prepare_value_nd(value, bcast_shape=bcast_shape, squeeze_axes=new_axes)
                value_nd = value_nd.reshape(indexed_shape)
                value_nd.copyto(self)
        elif isinstance(value, numeric_types):
            if isinstance(value, bool):
                self._full(int(value))
            else:
                self._full(value)
        elif isinstance(value, (np.ndarray, np.generic)):
            tmp_shape = _shape_for_bcast(value.shape, target_ndim=self.ndim, new_axes=int_axes)
            value = value.reshape(tmp_shape)
            if isinstance(value, np.generic) or value.shape != self.shape:
                value = np.broadcast_to(value, self.shape)
            self._sync_copyfrom(value)
        else:
            bcast_shape = self._drop_int_axes(indexed_shape, int_axes)
            value_nd = self._prepare_value_nd(value, bcast_shape=bcast_shape, squeeze_axes=new_axes)
            value_nd = value_nd.reshape(indexed_shape)
            value_nd.copyto(self)
    elif isinstance(value, numeric_types):
        self.slice_assign_scalar(float(value), begin, end, step)
    else:
        bcast_shape = self._drop_int_axes(indexed_shape, int_axes)
        value_nd = self._prepare_value_nd(value, bcast_shape=bcast_shape, squeeze_axes=new_axes)
        value_nd = value_nd.reshape(indexed_shape)
        self.slice_assign(value_nd, begin, end, step)
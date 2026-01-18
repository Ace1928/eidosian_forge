from array import array as native_array
import ctypes
import warnings
import numpy as _np
from ..autograd import is_recording
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _GRAD_REQ_MAP
from ..ndarray import indexing_key_expand_implicit_axes, get_indexing_dispatch_code,\
from ..ndarray._internal import _set_np_ndarray_class
from . import _op as _mx_np_op
from ..base import check_call, _LIB, NDArrayHandle, c_array
from ..base import mx_real_t, c_array_buf, mx_uint, numeric_types, integer_types
from ..context import Context
from ..util import set_module, wrap_np_unary_func, wrap_np_binary_func
from ..context import current_context
from ..ndarray import numpy as _mx_nd_np
from ..ndarray.numpy import _internal as _npi
from ..ndarray.ndarray import _storage_type, from_numpy
from .utils import _get_np_op
from .fallback import *  # pylint: disable=wildcard-import,unused-wildcard-import
from . import fallback
def _get_np_basic_indexing(self, key):
    """
        This function indexes ``self`` with a tuple of `slice` objects only.
        """
    key_nd = tuple((idx for idx in key if idx is not None))
    if len(key_nd) < self.ndim:
        raise RuntimeError('too few indices after normalization: expected `ndim` ({}) but got {}. This is a bug, please report it!'.format(self.ndim, len(key_nd)))
    if len(key_nd) > self.ndim:
        raise IndexError('too many indices ({}) for array with {} dimensions'.format(len(key_nd), self.ndim))
    none_axes = [ax for ax in range(len(key)) if key[ax] is None]
    slc_key, int_axes = self._basic_indexing_key_int_to_slice(key_nd)
    new_axes = self._new_axes_after_basic_indexing(none_axes, key)
    for ax in int_axes:
        if not -self.shape[ax] <= key_nd[ax] < self.shape[ax]:
            raise IndexError('index {} is out of bounds for axis {} with size {}'.format(key_nd[ax], ax, self.shape[ax]))
    if self._basic_indexing_slice_is_contiguous(slc_key, self.shape):
        flat_begin, flat_end = self._basic_indexing_contiguous_flat_begin_end(slc_key, self.shape)
        handle = NDArrayHandle()
        flat_self = self.reshape_view(-1)
        check_call(_LIB.MXNDArraySlice(flat_self.handle, mx_uint(flat_begin), mx_uint(flat_end), ctypes.byref(handle)))
        sliced_shape = self._basic_indexing_sliced_shape(slc_key, self.shape)
        sliced = self.__class__(handle=handle, writable=self.writable)
        if 0 in sliced_shape:
            sliced = sliced.reshape(sliced_shape)
        else:
            sliced = sliced.reshape_view(sliced_shape)
    else:
        begin, end, step = self._basic_indexing_key_to_begin_end_step(slc_key, self.shape, keep_none=True)
        sliced = _npi.slice(self, begin, end, step)
    final_shape = [sliced.shape[i] for i in range(sliced.ndim) if i not in int_axes]
    for ax in new_axes:
        final_shape.insert(ax, 1)
    if sliced.size == 0:
        return sliced.reshape(tuple(final_shape))
    else:
        return sliced.reshape_view(tuple(final_shape))
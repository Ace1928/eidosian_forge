import ctypes
import warnings
import operator
from array import array as native_array
import numpy as np
from ..base import NotSupportedForSparseNDArray
from ..base import _LIB, numeric_types
from ..base import c_array_buf, mx_real_t, integer_types
from ..base import NDArrayHandle, check_call
from ..context import Context, current_context
from . import _internal
from . import op
from ._internal import _set_ndarray_class
from .ndarray import NDArray, _storage_type, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ROW_SPARSE, _STORAGE_TYPE_CSR, _int64_enabled
from .ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray import zeros as _zeros_ndarray
from .ndarray import array as _array
from .ndarray import _ufunc_helper
def _csr_matrix_from_definition(data, indices, indptr, shape=None, ctx=None, dtype=None, indices_type=None, indptr_type=None):
    """Create a `CSRNDArray` based on data, indices and indptr"""
    storage_type = 'csr'
    ctx = current_context() if ctx is None else ctx
    dtype = _prepare_default_dtype(data, dtype)
    indptr_type = _STORAGE_AUX_TYPES[storage_type][0] if indptr_type is None else indptr_type
    indices_type = _STORAGE_AUX_TYPES[storage_type][1] if indices_type is None else indices_type
    data = _prepare_src_array(data, dtype)
    indptr = _prepare_src_array(indptr, indptr_type)
    indices = _prepare_src_array(indices, indices_type)
    if not isinstance(data, NDArray):
        data = _array(data, ctx, dtype)
    if not isinstance(indptr, NDArray):
        indptr = _array(indptr, ctx, indptr_type)
    if not isinstance(indices, NDArray):
        indices = _array(indices, ctx, indices_type)
    if shape is None:
        if indices.shape[0] == 0:
            raise ValueError('invalid shape')
        shape = (len(indptr) - 1, op.max(indices).asscalar() + 1)
    aux_shapes = [indptr.shape, indices.shape]
    if data.ndim != 1 or indptr.ndim != 1 or indices.ndim != 1 or (indptr.shape[0] == 0) or (len(shape) != 2):
        raise ValueError('invalid shape')
    result = CSRNDArray(_new_alloc_handle(storage_type, shape, ctx, False, dtype, [indptr_type, indices_type], aux_shapes))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, data.handle, ctypes.c_int(-1)))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, indptr.handle, ctypes.c_int(0)))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, indices.handle, ctypes.c_int(1)))
    return result
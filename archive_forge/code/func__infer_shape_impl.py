from array import array
import ctypes
import warnings
from numbers import Number
import numpy as _numpy  # pylint: disable=relative-import
from ..attribute import AttrScope
from ..base import _LIB, numeric_types, c_array, c_array_buf, c_str, c_str_array, c_handle_array
from ..base import mx_uint, py_str, string_types, integer_types, mx_int, mx_int64
from ..base import NDArrayHandle, ExecutorHandle, SymbolHandle
from ..base import check_call, MXNetError, NotImplementedForSymbol
from ..context import Context, current_context
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP, _GRAD_REQ_MAP
from ..ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, _int64_enabled, _SIGNED_INT32_UPPER_LIMIT
from ..ndarray import _ndarray_cls
from ..executor import Executor
from . import _internal
from . import op
from ._internal import SymbolBase, _set_symbol_class
from ..util import is_np_shape
def _infer_shape_impl(self, partial, *args, **kwargs):
    """The actual implementation for calling shape inference API."""
    if len(args) != 0 and len(kwargs) != 0:
        raise ValueError('Can only specify known argument                     shapes either by positional or kwargs way.')
    sdata = []
    indptr = [0]
    if len(args) != 0:
        keys = c_array(ctypes.c_char_p, [])
        for i, s in enumerate(args):
            if s is not None:
                if not isinstance(s, tuple):
                    raise TypeError('Arguments need to be shapes (tuple), but argument %d is %s.' % (i, type(s)))
                sdata.extend(s)
            indptr.append(len(sdata))
    else:
        str_keys = []
        for k, v in kwargs.items():
            if not isinstance(v, tuple):
                raise TypeError("Arguments need to be shapes (tuple), but '%s' is %s." % (k, type(v)))
            str_keys.append(k)
            sdata.extend(v)
            indptr.append(len(sdata))
        keys = c_str_array(str_keys)
    arg_shape_size = mx_uint()
    arg_shape_ndim = ctypes.POINTER(mx_int)()
    out_shape_size = mx_uint()
    out_shape_ndim = ctypes.POINTER(mx_int)()
    aux_shape_size = mx_uint()
    aux_shape_ndim = ctypes.POINTER(mx_int)()
    complete = ctypes.c_int()
    if _int64_enabled():
        arg_shape_data = ctypes.POINTER(ctypes.POINTER(mx_int64))()
        out_shape_data = ctypes.POINTER(ctypes.POINTER(mx_int64))()
        aux_shape_data = ctypes.POINTER(ctypes.POINTER(mx_int64))()
        if partial:
            infer_func = _LIB.MXSymbolInferShapePartialEx64
        else:
            infer_func = _LIB.MXSymbolInferShapeEx64
        check_call(infer_func(self.handle, mx_uint(len(indptr) - 1), keys, c_array_buf(mx_int64, array('q', indptr)), c_array_buf(mx_int64, array('q', sdata)), ctypes.byref(arg_shape_size), ctypes.byref(arg_shape_ndim), ctypes.byref(arg_shape_data), ctypes.byref(out_shape_size), ctypes.byref(out_shape_ndim), ctypes.byref(out_shape_data), ctypes.byref(aux_shape_size), ctypes.byref(aux_shape_ndim), ctypes.byref(aux_shape_data), ctypes.byref(complete)))
    else:
        for size in sdata:
            if size > _SIGNED_INT32_UPPER_LIMIT:
                raise Exception('[_infer_shape_impl] Size of tensor you are trying to ' + 'allocate is larger than 2^31 elements. Please build ' + 'with flag USE_INT64_TENSOR_SIZE=1')
        arg_shape_data = ctypes.POINTER(ctypes.POINTER(mx_int))()
        out_shape_data = ctypes.POINTER(ctypes.POINTER(mx_int))()
        aux_shape_data = ctypes.POINTER(ctypes.POINTER(mx_int))()
        if partial:
            infer_func = _LIB.MXSymbolInferShapePartialEx
        else:
            infer_func = _LIB.MXSymbolInferShapeEx
        check_call(infer_func(self.handle, mx_uint(len(indptr) - 1), keys, c_array_buf(mx_uint, array('I', indptr)), c_array_buf(mx_int, array('i', sdata)), ctypes.byref(arg_shape_size), ctypes.byref(arg_shape_ndim), ctypes.byref(arg_shape_data), ctypes.byref(out_shape_size), ctypes.byref(out_shape_ndim), ctypes.byref(out_shape_data), ctypes.byref(aux_shape_size), ctypes.byref(aux_shape_ndim), ctypes.byref(aux_shape_data), ctypes.byref(complete)))
    if complete.value != 0:
        arg_shapes = [tuple(arg_shape_data[i][:arg_shape_ndim[i]]) if arg_shape_ndim[i] >= 0 else None for i in range(arg_shape_size.value)]
        out_shapes = [tuple(out_shape_data[i][:out_shape_ndim[i]]) if out_shape_ndim[i] >= 0 else None for i in range(out_shape_size.value)]
        aux_shapes = [tuple(aux_shape_data[i][:aux_shape_ndim[i]]) if aux_shape_ndim[i] >= 0 else None for i in range(aux_shape_size.value)]
        return (arg_shapes, out_shapes, aux_shapes)
    else:
        return (None, None, None)
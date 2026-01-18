import traceback
import warnings
import collections
from array import array
from threading import Lock
import ctypes
from ctypes import CFUNCTYPE, POINTER, Structure, pointer
from ctypes import c_void_p, c_int, c_char, c_char_p, cast, c_bool
from .base import _LIB, check_call, MXCallbackList, c_array, c_array_buf, mx_int, OpHandle
from .base import c_str, mx_uint, mx_float, ctypes2numpy_shared, NDArrayHandle, py_str
from . import symbol, context
from .ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ID_TO_STR
from .ndarray.ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray.ndarray import _STORAGE_TYPE_CSR, _STORAGE_TYPE_ROW_SPARSE
from .ndarray import _ndarray_cls
from .numpy.multiarray import _np_ndarray_cls
from .util import is_np_array
class NDArrayOpInfo(Structure):
    """Structure that holds Callback information. Passed to NDArrayOpProp"""
    _fields_ = [('forward', fb_functype), ('backward', fb_functype), ('infer_shape', infer_functype), ('list_outputs', list_functype), ('list_arguments', list_functype), ('declare_backward_dependency', deps_functype), ('p_forward', c_void_p), ('p_backward', c_void_p), ('p_infer_shape', c_void_p), ('p_list_outputs', c_void_p), ('p_list_arguments', c_void_p), ('p_declare_backward_dependency', c_void_p)]
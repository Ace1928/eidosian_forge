import ctypes
from ...base import NDArrayHandle
from ... import _global_var
class MXNetValue(ctypes.Union):
    """MXNetValue in C API"""
    _fields_ = [('v_int64', ctypes.c_int64), ('v_float64', ctypes.c_double), ('v_handle', ctypes.c_void_p), ('v_str', ctypes.c_char_p)]
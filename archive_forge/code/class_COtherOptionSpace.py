import ctypes
from ..base import _LIB
from ..base import c_str_array, c_array
from ..base import check_call
class COtherOptionSpace(ctypes.Structure):
    """ctypes data structure for OtherOptionSpace"""
    _fields_ = [('entities', ctypes.POINTER(COtherOptionEntity)), ('entities_size', ctypes.c_int)]
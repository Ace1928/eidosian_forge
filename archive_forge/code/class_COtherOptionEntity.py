import ctypes
from ..base import _LIB
from ..base import c_str_array, c_array
from ..base import check_call
class COtherOptionEntity(ctypes.Structure):
    """ctypes data structure for OtherOptionEntity"""
    _fields_ = [('val', ctypes.c_int)]
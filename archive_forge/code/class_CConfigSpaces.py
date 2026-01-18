import ctypes
from ..base import _LIB
from ..base import c_str_array, c_array
from ..base import check_call
class CConfigSpaces(ctypes.Structure):
    """ctypes data structure for ConfigSpaces"""
    _fields_ = [('spaces_size', ctypes.c_int), ('spaces_key', ctypes.POINTER(ctypes.c_char_p)), ('spaces_val', ctypes.POINTER(CConfigSpace))]
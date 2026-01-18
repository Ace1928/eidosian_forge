import ctypes
from ..base import _LIB
from ..base import c_str_array, c_array
from ..base import check_call
class CConfigSpace(ctypes.Structure):
    """ctypes data structure for ConfigSpace"""
    _fields_ = [('entity_map_size', ctypes.c_int), ('entity_map_key', ctypes.POINTER(ctypes.c_char_p)), ('entity_map_val', ctypes.POINTER(COtherOptionEntity)), ('space_map_size', ctypes.c_int), ('space_map_key', ctypes.POINTER(ctypes.c_char_p)), ('space_map_val', ctypes.POINTER(COtherOptionSpace))]
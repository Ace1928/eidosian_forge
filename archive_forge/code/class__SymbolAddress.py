import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint64, c_size_t
from llvmlite.binding import ffi, targets
class _SymbolAddress(ctypes.Structure):
    _fields_ = [('name', c_char_p), ('address', c_uint64)]
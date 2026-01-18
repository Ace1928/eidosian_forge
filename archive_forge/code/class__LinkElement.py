import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint64, c_size_t
from llvmlite.binding import ffi, targets
class _LinkElement(ctypes.Structure):
    _fields_ = [('element_kind', c_uint8), ('value', c_char_p), ('value_len', c_size_t)]
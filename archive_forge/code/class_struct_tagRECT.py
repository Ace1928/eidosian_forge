from ctypes import *
from ctypes import _SimpleCData, _check_size
from OpenGL import extensions
from OpenGL.raw.GL._types import *
from OpenGL._bytes import as_8_bit
from OpenGL._opaque import opaque_pointer_cls as _opaque_pointer_cls
class struct_tagRECT(Structure):
    __slots__ = ['left', 'top', 'right', 'bottom']
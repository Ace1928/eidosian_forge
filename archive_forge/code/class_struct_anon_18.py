from OpenGL import platform as _p, constant, extensions
from ctypes import *
from OpenGL.raw.GL._types import *
from OpenGL._bytes import as_8_bit
class struct_anon_18(Structure):
    __slots__ = ['ext_data', 'visualid', 'class', 'red_mask', 'green_mask', 'blue_mask', 'bits_per_rgb', 'map_entries']
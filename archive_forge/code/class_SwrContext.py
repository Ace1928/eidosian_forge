from ctypes import c_int, c_int64
from ctypes import c_uint8
from ctypes import c_void_p, POINTER, Structure
import pyglet.lib
from pyglet.util import debug_print
from . import compat
class SwrContext(Structure):
    pass
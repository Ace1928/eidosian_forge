from ctypes import c_int, c_uint16, c_int64, c_uint32, c_uint64
from ctypes import c_uint8, c_uint, c_float, c_char_p
from ctypes import c_void_p, POINTER, CFUNCTYPE, Structure
import pyglet.lib
from pyglet.util import debug_print
from . import compat
from . import libavutil
class AVCodec(Structure):
    _fields_ = [('name', c_char_p), ('long_name', c_char_p), ('type', c_int), ('id', c_int), ('capabilities', c_int), ('supported_framerates', POINTER(AVRational)), ('pix_fmts', POINTER(c_int)), ('supported_samplerates', POINTER(c_int)), ('sample_fmts', POINTER(c_int)), ('channel_layouts', POINTER(c_uint64)), ('max_lowres', c_uint8)]
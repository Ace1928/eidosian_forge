from OpenGL import platform as _p, constant, extensions
from ctypes import *
from OpenGL.raw.GL._types import *
from OpenGL._bytes import as_8_bit
class struct_anon_112(Structure):
    __slots__ = ['type', 'serial', 'send_event', 'display', 'drawable', 'event_type', 'ust', 'msc', 'sbc']
import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_buffer_attr(Structure):
    __slots__ = ['maxlength', 'tlength', 'prebuf', 'minreq', 'fragsize']
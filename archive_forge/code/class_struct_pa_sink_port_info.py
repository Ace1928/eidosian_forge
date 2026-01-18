import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_sink_port_info(Structure):
    __slots__ = ['name', 'description', 'priority', 'available']
import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_channel_map(Structure):
    __slots__ = ['channels', 'map']
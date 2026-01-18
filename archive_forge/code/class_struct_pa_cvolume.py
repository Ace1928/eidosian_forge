import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_cvolume(Structure):
    __slots__ = ['channels', 'values']
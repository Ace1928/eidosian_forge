import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_sample_spec(Structure):
    __slots__ = ['format', 'rate', 'channels']
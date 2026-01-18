import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_sample_info(Structure):
    __slots__ = ['index', 'name', 'volume', 'sample_spec', 'channel_map', 'duration', 'bytes', 'lazy', 'filename', 'proplist']
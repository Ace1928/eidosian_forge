import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_module_info(Structure):
    __slots__ = ['index', 'name', 'argument', 'n_used', 'auto_unload', 'proplist']
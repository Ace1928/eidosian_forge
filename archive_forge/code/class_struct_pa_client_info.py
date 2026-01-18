import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_client_info(Structure):
    __slots__ = ['index', 'name', 'owner_module', 'driver', 'proplist']
import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_spawn_api(Structure):
    __slots__ = ['prefork', 'postfork', 'atfork']
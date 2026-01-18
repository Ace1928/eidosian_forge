import ctypes
from ctypes import *
import pyglet.lib
import pyglet.libs.x11.xlib
class struct__XKeyInfo(Structure):
    __slots__ = ['class', 'length', 'min_keycode', 'max_keycode', 'num_keys']
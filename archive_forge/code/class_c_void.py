import ctypes
from ctypes import *
import pyglet.lib
class c_void(Structure):
    _fields_ = [('dummy', c_int)]
import ctypes
from ctypes import *
import pyglet.lib
class struct_funcs(Structure):
    __slots__ = ['create_image', 'destroy_image', 'get_pixel', 'put_pixel', 'sub_image', 'add_pixel']
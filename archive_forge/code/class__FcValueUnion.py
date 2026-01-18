from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
class _FcValueUnion(Union):
    _fields_ = [('s', c_char_p), ('i', c_int), ('b', c_int), ('d', c_double), ('m', c_void_p), ('c', c_void_p), ('f', c_void_p), ('p', c_void_p), ('l', c_void_p)]
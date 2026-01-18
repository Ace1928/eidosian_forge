from ctypes import *
from .base import FontException
import pyglet.lib
class FT_Generic(Structure):
    _fields_ = [('data', c_void_p), ('finalizer', FT_Generic_Finalizer)]
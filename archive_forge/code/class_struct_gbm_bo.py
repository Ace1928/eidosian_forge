from ctypes import *
import pyglet.lib
class struct_gbm_bo(Structure):
    _fields_ = [('_opaque_struct', c_int)]
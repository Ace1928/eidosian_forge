from ctypes import *
import pyglet.lib
class struct_gbm_format_name_desc(Structure):
    _fields_ = [('_opaque_struct', c_int)]
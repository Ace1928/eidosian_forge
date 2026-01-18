from ctypes import *
import pyglet.lib
def __gbm_fourcc_code(a, b, c, d):
    a, b, c, d = (ord(a), ord(b), ord(c), ord(d))
    return c_uint32(a).value | c_uint32(b).value << 8 | c_uint32(c).value << 16 | c_uint32(d).value << 24
from ctypes import c_void_p, c_int, c_bool, Structure, c_uint32, util, cdll, c_uint, c_double, POINTER, c_int64, \
from pyglet.libs.darwin import CFURLRef
class AudioBufferList(Structure):
    _fields_ = [('mNumberBuffers', c_uint), ('mBuffers', AudioBuffer * 1)]
from ctypes import c_void_p, c_int, c_bool, Structure, c_uint32, util, cdll, c_uint, c_double, POINTER, c_int64, \
from pyglet.libs.darwin import CFURLRef
class AudioBuffer(Structure):
    _fields_ = [('mNumberChannels', c_uint), ('mDataByteSize', c_uint), ('mData', c_void_p)]
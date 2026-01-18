from ctypes import c_void_p, c_int, c_bool, Structure, c_uint32, util, cdll, c_uint, c_double, POINTER, c_int64, \
from pyglet.libs.darwin import CFURLRef
class AudioStreamPacketDescription(Structure):
    _fields_ = [('mStartOffset', c_int64), ('mVariableFramesInPacket', c_uint32), ('mDataByteSize', c_uint32)]
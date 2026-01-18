import ctypes
class ApplyResultObj(ctypes.c_void_p):

    def __init__(self, obj):
        self._as_parameter_ = obj

    def from_param(obj):
        return obj
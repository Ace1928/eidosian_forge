import ctypes
class RCFNumObj(ctypes.c_void_p):

    def __init__(self, e):
        self._as_parameter_ = e

    def from_param(obj):
        return obj
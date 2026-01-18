import ctypes
class FuncInterpObj(ctypes.c_void_p):

    def __init__(self, f):
        self._as_parameter_ = f

    def from_param(obj):
        return obj
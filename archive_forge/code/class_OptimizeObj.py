import ctypes
class OptimizeObj(ctypes.c_void_p):

    def __init__(self, optimize):
        self._as_parameter_ = optimize

    def from_param(obj):
        return obj
import ctypes
class FixedpointObj(ctypes.c_void_p):

    def __init__(self, fixedpoint):
        self._as_parameter_ = fixedpoint

    def from_param(obj):
        return obj
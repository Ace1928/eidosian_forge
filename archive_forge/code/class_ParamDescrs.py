import ctypes
class ParamDescrs(ctypes.c_void_p):

    def __init__(self, paramdescrs):
        self._as_parameter_ = paramdescrs

    def from_param(obj):
        return obj
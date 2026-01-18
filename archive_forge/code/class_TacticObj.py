import ctypes
class TacticObj(ctypes.c_void_p):

    def __init__(self, tactic):
        self._as_parameter_ = tactic

    def from_param(obj):
        return obj
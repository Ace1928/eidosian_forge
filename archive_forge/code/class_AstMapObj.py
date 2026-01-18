import ctypes
class AstMapObj(ctypes.c_void_p):

    def __init__(self, ast_map):
        self._as_parameter_ = ast_map

    def from_param(obj):
        return obj
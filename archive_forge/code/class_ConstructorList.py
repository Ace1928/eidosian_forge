import ctypes
class ConstructorList(ctypes.c_void_p):

    def __init__(self, constructor_list):
        self._as_parameter_ = constructor_list

    def from_param(obj):
        return obj
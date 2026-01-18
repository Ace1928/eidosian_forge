import ctypes, ctypes.util, operator, sys
from . import model
class CTypesLibrary(object):

    def __init__(self, backend, cdll):
        self.backend = backend
        self.cdll = cdll

    def load_function(self, BType, name):
        c_func = getattr(self.cdll, name)
        funcobj = BType._from_ctypes(c_func)
        funcobj._name = name
        return funcobj

    def read_variable(self, BType, name):
        try:
            ctypes_obj = BType._ctype.in_dll(self.cdll, name)
        except AttributeError as e:
            raise NotImplementedError(e)
        return BType._from_ctypes(ctypes_obj)

    def write_variable(self, BType, name, value):
        new_ctypes_obj = BType._to_ctypes(value)
        ctypes_obj = BType._ctype.in_dll(self.cdll, name)
        ctypes.memmove(ctypes.addressof(ctypes_obj), ctypes.addressof(new_ctypes_obj), ctypes.sizeof(BType._ctype))
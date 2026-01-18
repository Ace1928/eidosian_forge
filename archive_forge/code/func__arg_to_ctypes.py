import ctypes, ctypes.util, operator, sys
from . import model
@classmethod
def _arg_to_ctypes(cls, *value):
    if value and isinstance(value[0], bytes):
        return ctypes.c_char_p(value[0])
    else:
        return super(CTypesPtr, cls)._arg_to_ctypes(*value)
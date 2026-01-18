import ctypes, ctypes.util, operator, sys
from . import model
@classmethod
def _new_pointer_at(cls, address):
    self = cls.__new__(cls)
    self._address = address
    self._as_ctype_ptr = ctypes.cast(address, cls._ctype)
    return self
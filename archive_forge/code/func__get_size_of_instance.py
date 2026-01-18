import ctypes, ctypes.util, operator, sys
from . import model
def _get_size_of_instance(self):
    return ctypes.sizeof(self._ctype)
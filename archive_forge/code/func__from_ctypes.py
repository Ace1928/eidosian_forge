import ctypes, ctypes.util, operator, sys
from . import model
@staticmethod
def _from_ctypes(ctypes_array):
    self = CTypesArray.__new__(CTypesArray)
    self._blob = ctypes_array
    return self
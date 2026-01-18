import sys
import ctypes
from pyglet.util import debug_print
def AddRef(self):
    self._vrefcount += 1
    return self._vrefcount
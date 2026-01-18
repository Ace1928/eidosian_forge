import sys
import ctypes
from pyglet.util import debug_print
def Release(self):
    if self._vrefcount <= 0:
        assert _debug_com(f'COMObject {self}: Release while refcount was {self._vrefcount}')
    self._vrefcount -= 1
    return self._vrefcount
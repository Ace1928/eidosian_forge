import ctypes
from .base import PlatformEventLoop
from pyglet.libs.win32 import _kernel32, _user32, types, constants
from pyglet.libs.win32.types import *
def _recreate_wait_objects_array(self):
    if not self._wait_objects:
        self._wait_objects_n = 0
        self._wait_objects_array = None
        return
    self._wait_objects_n = len(self._wait_objects)
    self._wait_objects_array = (HANDLE * self._wait_objects_n)(*[o for o, f in self._wait_objects])
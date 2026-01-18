import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
def _set_float(self, key, value):
    al.alListenerf(key, float(value))
    self._check_error('Failed to set value.')
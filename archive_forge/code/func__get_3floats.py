import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
def _get_3floats(self, key):
    x = al.ALfloat()
    y = al.ALfloat()
    z = al.ALfloat()
    al.alGetListener3f(key, x, y, z)
    self._check_error('Failed to get value')
    return (x.value, y.value, z.value)
import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
def _set_float_vector(self, key, values):
    al_float_vector = (al.ALfloat * len(values))(*values)
    al.alListenerfv(key, al_float_vector)
    self._check_error('Failed to set value.')
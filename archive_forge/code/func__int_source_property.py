import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
def _int_source_property(attribute):
    return property(lambda self: self._get_int(attribute), lambda self, value: self._set_int(attribute, value))
from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
def _set_double(self, name, value):
    assert self._pattern
    assert name
    assert self._fontconfig
    if not value:
        return
    self._fontconfig.FcPatternAddDouble(self._pattern, name, c_double(value))
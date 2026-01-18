from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
def _get_double(self, name):
    value = self._get_value(name)
    if value and value.type == FcTypeDouble:
        return value.u.d
    else:
        return None
from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
def _prepare_search_pattern(self):
    self._create()
    self._set_string(FC_FAMILY, self.name)
    self._set_double(FC_SIZE, self.size)
    self._set_integer(FC_WEIGHT, self._bold_to_weight(self.bold))
    self._set_integer(FC_SLANT, self._italic_to_slant(self.italic))
    self._substitute_defaults()
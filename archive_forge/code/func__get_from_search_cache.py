from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
def _get_from_search_cache(self, name, size, bold, italic):
    result = self._search_cache.get((name, size, bold, italic), None)
    if result and result.is_valid:
        return result
    else:
        return None
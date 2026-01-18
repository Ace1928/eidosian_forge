from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
def _add_to_search_cache(self, search_pattern, result_pattern):
    self._search_cache[search_pattern.name, search_pattern.size, search_pattern.bold, search_pattern.italic] = result_pattern
    if len(self._search_cache) > self._cache_size:
        self._search_cache.popitem(last=False)[1].dispose()
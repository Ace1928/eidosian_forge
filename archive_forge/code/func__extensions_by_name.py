import logging
import operator
from . import _cache
from .exception import NoMatches
@property
def _extensions_by_name(self):
    if self._extensions_by_name_cache is None:
        d = {}
        for e in self.extensions:
            d[e.name] = e
        self._extensions_by_name_cache = d
    return self._extensions_by_name_cache
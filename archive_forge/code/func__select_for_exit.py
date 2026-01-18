from __future__ import absolute_import
import itertools
import sys
from weakref import ref
@classmethod
def _select_for_exit(cls):
    L = [(f, i) for f, i in cls._registry.items() if i.atexit]
    L.sort(key=lambda item: item[1].index)
    return [f for f, i in L]
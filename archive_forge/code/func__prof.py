import sys
import types as Types
import warnings
import weakref as Weakref
from inspect import isbuiltin, isclass, iscode, isframe, isfunction, ismethod, ismodule
from math import log
from os import curdir, linesep
from struct import calcsize
from gc import get_objects as _getobjects
from gc import get_referents as _getreferents  # containers only?
from array import array as _array  # array type
def _prof(self, key):
    """Get _Prof object."""
    p = self._profs.get(key, None)
    if not p:
        self._profs[key] = p = _Prof()
        self.exclude_objs(p)
    return p
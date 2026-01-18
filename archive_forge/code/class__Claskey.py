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
class _Claskey(object):
    """Wrapper for class objects."""
    __slots__ = ('_obj',)

    def __init__(self, obj):
        self._obj = obj

    def __str__(self):
        r = str(self._obj)
        return r[:-1] + ' def>' if r.endswith('>') else r + ' def'
    __repr__ = __str__
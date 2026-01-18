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
def _iter_refs(obj, named):
    """Return the referent(s) of an iterator object."""
    r = _getreferents(obj)
    return _refs(r, named, itor=_nameof(obj) or 'iteref')
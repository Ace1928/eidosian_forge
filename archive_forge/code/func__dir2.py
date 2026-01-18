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
def _dir2(obj, pref=_NN, excl=(), slots=None, itor=_NN):
    """Return an attribute name, object 2-tuple for certain
    attributes or for the ``__slots__`` attributes of the
    given object, but not both.  Any iterator referent
    objects are returned with the given name if the
    latter is non-empty.
    """
    if slots:
        if hasattr(obj, slots):
            s = {}
            for c in type(obj).mro():
                n = _nameof(c)
                for a in getattr(c, slots, ()):
                    if a.startswith('__'):
                        a = '_' + n + a
                    if hasattr(obj, a):
                        s.setdefault(a, getattr(obj, a))
            for t in _items(s):
                yield t
    elif itor:
        for o in obj:
            yield (itor, o)
    else:
        for a in dir(obj):
            if a.startswith(pref) and hasattr(obj, a) and (a not in excl):
                yield (a, getattr(obj, a))
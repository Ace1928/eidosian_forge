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
def _infer_dict(obj):
    """Return True for likely dict object via duck typing."""
    for attrs in (('items', 'keys', 'values'), ('iteritems', 'iterkeys', 'itervalues')):
        attrs += ('__len__', 'get', 'has_key')
        if all((callable(getattr(obj, a, None)) for a in attrs)):
            return True
    return False
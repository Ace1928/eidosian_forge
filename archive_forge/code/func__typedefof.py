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
def _typedefof(obj, save=False, **opts):
    """Get the typedef for an object."""
    k = _objkey(obj)
    v = _typedefs.get(k, None)
    if not v:
        v = _typedef(obj, **opts)
        if save:
            _typedefs[k] = v
    return v
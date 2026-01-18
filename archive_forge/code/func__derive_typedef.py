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
def _derive_typedef(typ):
    """Return single, existing super type typedef or None."""
    v = [v for v in _values(_typedefs) if _issubclass(typ, v.type)]
    return v[0] if len(v) == 1 else None
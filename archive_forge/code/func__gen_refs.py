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
def _gen_refs(obj, named):
    """Return the referent(s) of a generator (expression) object."""
    f = getattr(obj, 'gi_frame', None)
    return _refs(f, named, 'f_locals', 'f_code')
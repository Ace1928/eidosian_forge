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
def _len_set(obj):
    """Length of frozen/set (estimate)."""
    n = len(obj)
    if n > 8:
        n = _power_of_2(n + n - 2)
    elif n:
        n = 8
    return n
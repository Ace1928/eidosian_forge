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
def _len_dict(obj):
    """Dict length in items (estimate)."""
    n = len(obj)
    if n < 6:
        n = 0
    else:
        n = _power_of_2(n + 1)
    return n
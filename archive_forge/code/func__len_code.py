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
def _len_code(obj):
    """Length of code object (stack and variables only)."""
    return _len(obj.co_freevars) + obj.co_stacksize + _len(obj.co_cellvars) + obj.co_nlocals - 1
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
def _power_of_2(n):
    """Find the next power of 2."""
    p2 = 2 ** int(log(n, 2))
    while n > p2:
        p2 += p2
    return p2
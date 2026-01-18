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
def _p100(part, total, prec=1):
    """Return percentage as string."""
    t = float(total)
    if t > 0:
        p = part * 100.0 / t
        r = '%.*f%%' % (prec, p)
    else:
        r = 'n/a'
    return r
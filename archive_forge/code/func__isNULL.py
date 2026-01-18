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
def _isNULL(obj):
    """Prevent asizeof(all=True, ...) crash.

    Sizing gc.get_objects() crashes in Pythonista3 with
    Python 3.5.1 on iOS due to 1-tuple (<Null>,) object,
    see <http://forum.omz-software.com/user/mrjean1>.
    """
    return isinstance(obj, tuple) and len(obj) == 1 and (repr(obj) == '(<NULL>,)')
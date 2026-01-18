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
def _objs_opts_x(where, objs, all=None, **opts):
    """Return the given or 'all' objects plus
    the remaining options and exclude flag
    """
    if objs:
        t, x = (objs, False)
    elif all in (False, None):
        t, x = ((), True)
    elif all is True:
        t, x = (_getobjects(), True)
    else:
        raise _OptionError(where, all=all)
    return (t, opts, x)
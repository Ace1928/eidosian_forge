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
def _sizes(self, objs, sized=None):
    """Return the size or an **Asized** instance for each
        given object plus the total size.  The total includes
        the size of duplicates only once.
        """
    self.exclude_refs(*objs)
    s, t = ({}, [])
    self.exclude_objs(s, t)
    for o in objs:
        i = id(o)
        if i in s:
            self._seen.again(i)
        else:
            s[i] = self._sizer(o, 0, 0, sized)
        t.append(s[i])
    return tuple(t)
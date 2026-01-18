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
class _Prof(object):
    """Internal type profile class."""
    high = 0
    number = 0
    objref = None
    total = 0
    weak = False

    def __cmp__(self, other):
        if self.total < other.total:
            return -1
        elif self.total > other.total:
            return +1
        elif self.number < other.number:
            return -1
        elif self.number > other.number:
            return +1
        return 0

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def format(self, clip=0, grand=None):
        """Return format dict."""
        if self.number > 1:
            a, p = (int(self.total / self.number), 's')
        else:
            a, p = (self.total, _NN)
        o = self.objref
        if self.weak:
            o = o()
        t = _SI2(self.total)
        if grand:
            t += ' (%s)' % _p100(self.total, grand, prec=0)
        return dict(avg=_SI2(a), high=_SI2(self.high), lengstr=_lengstr(o), obj=_repr(o, clip=clip), plural=p, total=t)

    def update(self, obj, size):
        """Update this profile."""
        self.number += 1
        self.total += size
        if self.high < size:
            self.high = size
            try:
                self.objref, self.weak = (Weakref.ref(obj), True)
            except TypeError:
                self.objref, self.weak = (obj, False)
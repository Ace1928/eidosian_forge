import copy
import datetime
import decimal
import types
import warnings
from functools import reduce
from zope.interface import implementer
from incremental import Version
from twisted.persisted.crefutil import (
from twisted.python.compat import nativeString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import namedAny, namedObject, qual
from twisted.spread.interfaces import IJellyable, IUnjellyable
def _unjelly_tuple(self, lst):
    l = list(range(len(lst)))
    finished = 1
    for elem in l:
        if isinstance(self.unjellyInto(l, elem, lst[elem]), NotKnown):
            finished = 0
    if finished:
        return tuple(l)
    else:
        return _Tuple(l)
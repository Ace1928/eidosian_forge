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
def _unjelly_dereference(self, lst):
    refid = lst[0]
    x = self.references.get(refid)
    if x is not None:
        return x
    der = _Dereference(refid)
    self.references[refid] = der
    return der
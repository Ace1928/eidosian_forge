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
def _unjelly_dictionary(self, lst):
    d = {}
    for k, v in lst:
        kvd = _DictKeyAndValue(d)
        self.unjellyInto(kvd, 0, k)
        self.unjellyInto(kvd, 1, v)
    return d
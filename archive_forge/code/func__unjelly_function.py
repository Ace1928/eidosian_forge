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
def _unjelly_function(self, rest):
    fname = nativeString(rest[0])
    modSplit = fname.split(nativeString('.'))
    modName = nativeString('.').join(modSplit[:-1])
    if not self.taster.isModuleAllowed(modName):
        raise InsecureJelly('Module not allowed: %s' % modName)
    function = namedAny(fname)
    return function
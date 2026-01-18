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
def _unjelly_module(self, rest):
    moduleName = nativeString(rest[0])
    if type(moduleName) != str:
        raise InsecureJelly('Attempted to unjelly a module with a non-string name.')
    if not self.taster.isModuleAllowed(moduleName):
        raise InsecureJelly(f'Attempted to unjelly module named {moduleName!r}')
    mod = __import__(moduleName, {}, {}, 'x')
    return mod
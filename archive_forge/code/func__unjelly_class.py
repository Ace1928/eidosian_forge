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
def _unjelly_class(self, rest):
    cname = nativeString(rest[0])
    clist = cname.split(nativeString('.'))
    modName = nativeString('.').join(clist[:-1])
    if not self.taster.isModuleAllowed(modName):
        raise InsecureJelly('module %s not allowed' % modName)
    klaus = namedObject(cname)
    objType = type(klaus)
    if objType is not type:
        raise InsecureJelly("class %r unjellied to something that isn't a class: %r" % (cname, klaus))
    if not self.taster.isClassAllowed(klaus):
        raise InsecureJelly('class not allowed: %s' % qual(klaus))
    return klaus
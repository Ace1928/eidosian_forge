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
def _createBlank(cls):
    """
    Given an object, if that object is a type, return a new, blank instance
    of that type which has not had C{__init__} called on it.  If the object
    is not a type, return L{None}.

    @param cls: The type (or class) to create an instance of.
    @type cls: L{type} or something else that cannot be
        instantiated.

    @return: a new blank instance or L{None} if C{cls} is not a class or type.
    """
    if isinstance(cls, type):
        return cls.__new__(cls)
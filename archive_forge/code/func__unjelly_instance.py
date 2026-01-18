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
def _unjelly_instance(self, rest):
    """
        (internal) Unjelly an instance.

        Called to handle the deprecated I{instance} token.

        @param rest: The s-expression representing the instance.

        @return: The unjellied instance.
        """
    warnings.warn_explicit('Unjelly support for the instance atom is deprecated since Twisted 15.0.0.  Upgrade peer for modern instance support.', category=DeprecationWarning, filename='', lineno=0)
    clz = self.unjelly(rest[0])
    return self._genericUnjelly(clz, rest[1])
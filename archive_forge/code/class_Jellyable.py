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
@implementer(IJellyable)
class Jellyable:
    """
    Inherit from me to Jelly yourself directly with the `getStateFor'
    convenience method.
    """

    def getStateFor(self, jellier):
        return self.__dict__

    def jellyFor(self, jellier):
        """
        @see: L{twisted.spread.interfaces.IJellyable.jellyFor}
        """
        sxp = jellier.prepare(self)
        sxp.extend([qual(self.__class__).encode('utf-8'), jellier.jelly(self.getStateFor(jellier))])
        return jellier.preserve(self, sxp)
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
@implementer(IUnjellyable)
class Unjellyable:
    """
    Inherit from me to Unjelly yourself directly with the
    C{setStateFor} convenience method.
    """

    def setStateFor(self, unjellier, state):
        self.__dict__ = state

    def unjellyFor(self, unjellier, jellyList):
        """
        Perform the inverse operation of L{Jellyable.jellyFor}.

        @see: L{twisted.spread.interfaces.IUnjellyable.unjellyFor}
        """
        state = unjellier.unjelly(jellyList[1])
        self.setStateFor(unjellier, state)
        return self
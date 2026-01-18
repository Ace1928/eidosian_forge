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
def _unjellySetOrFrozenset(self, lst, containerType):
    """
        Helper method to unjelly set or frozenset.

        @param lst: the content of the set.
        @type lst: C{list}

        @param containerType: the type of C{set} to use.
        """
    l = list(range(len(lst)))
    finished = True
    for elem in l:
        data = self.unjellyInto(l, elem, lst[elem])
        if isinstance(data, NotKnown):
            finished = False
    if not finished:
        return _Container(l, containerType)
    else:
        return containerType(l)
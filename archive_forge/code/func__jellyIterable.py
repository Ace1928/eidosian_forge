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
def _jellyIterable(self, atom, obj):
    """
        Jelly an iterable object.

        @param atom: the identifier atom of the object.
        @type atom: C{str}

        @param obj: any iterable object.
        @type obj: C{iterable}

        @return: a generator of jellied data.
        @rtype: C{generator}
        """
    yield atom
    for item in obj:
        yield self.jelly(item)
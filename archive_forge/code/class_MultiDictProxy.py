import sys
import types
from array import array
from collections import abc
from ._abc import MultiMapping, MutableMultiMapping
class MultiDictProxy(_Base, MultiMapping):
    """Read-only proxy for MultiDict instance."""

    def __init__(self, arg):
        if not isinstance(arg, (MultiDict, MultiDictProxy)):
            raise TypeError('ctor requires MultiDict or MultiDictProxy instance, not {}'.format(type(arg)))
        self._impl = arg._impl

    def __reduce__(self):
        raise TypeError("can't pickle {} objects".format(self.__class__.__name__))

    def copy(self):
        """Return a copy of itself."""
        return MultiDict(self.items())
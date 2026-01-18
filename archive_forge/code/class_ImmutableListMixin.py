from __future__ import annotations
from itertools import repeat
from .._internal import _missing
class ImmutableListMixin:
    """Makes a :class:`list` immutable.

    .. versionadded:: 0.5

    :private:
    """
    _hash_cache = None

    def __hash__(self):
        if self._hash_cache is not None:
            return self._hash_cache
        rv = self._hash_cache = hash(tuple(self))
        return rv

    def __reduce_ex__(self, protocol):
        return (type(self), (list(self),))

    def __delitem__(self, key):
        is_immutable(self)

    def __iadd__(self, other):
        is_immutable(self)

    def __imul__(self, other):
        is_immutable(self)

    def __setitem__(self, key, value):
        is_immutable(self)

    def append(self, item):
        is_immutable(self)

    def remove(self, item):
        is_immutable(self)

    def extend(self, iterable):
        is_immutable(self)

    def insert(self, pos, value):
        is_immutable(self)

    def pop(self, index=-1):
        is_immutable(self)

    def reverse(self):
        is_immutable(self)

    def sort(self, key=None, reverse=False):
        is_immutable(self)
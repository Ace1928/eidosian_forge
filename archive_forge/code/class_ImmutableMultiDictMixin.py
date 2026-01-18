from __future__ import annotations
from itertools import repeat
from .._internal import _missing
class ImmutableMultiDictMixin(ImmutableDictMixin):
    """Makes a :class:`MultiDict` immutable.

    .. versionadded:: 0.5

    :private:
    """

    def __reduce_ex__(self, protocol):
        return (type(self), (list(self.items(multi=True)),))

    def _iter_hashitems(self):
        return self.items(multi=True)

    def add(self, key, value):
        is_immutable(self)

    def popitemlist(self):
        is_immutable(self)

    def poplist(self, key):
        is_immutable(self)

    def setlist(self, key, new_list):
        is_immutable(self)

    def setlistdefault(self, key, default_list=None):
        is_immutable(self)
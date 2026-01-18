import collections
from typing import Any, Set
import weakref
class ObjectIdentityWeakSet(ObjectIdentitySet):
    """Like weakref.WeakSet, but compares objects with "is"."""
    __slots__ = ()

    def _wrap_key(self, key):
        return _WeakObjectIdentityWrapper(key)

    def __len__(self):
        return len([_ for _ in self])

    def __iter__(self):
        keys = list(self._storage)
        for key in keys:
            unwrapped = key.unwrapped
            if unwrapped is None:
                self.discard(key)
            else:
                yield unwrapped
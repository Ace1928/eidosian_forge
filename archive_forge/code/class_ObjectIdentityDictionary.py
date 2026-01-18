import collections
from typing import Any, Set
import weakref
class ObjectIdentityDictionary(collections.abc.MutableMapping):
    """A mutable mapping data structure which compares using "is".

  This is necessary because we have trackable objects (_ListWrapper) which
  have behavior identical to built-in Python lists (including being unhashable
  and comparing based on the equality of their contents by default).
  """
    __slots__ = ['_storage']

    def __init__(self):
        self._storage = {}

    def _wrap_key(self, key):
        return _ObjectIdentityWrapper(key)

    def __getitem__(self, key):
        return self._storage[self._wrap_key(key)]

    def __setitem__(self, key, value):
        self._storage[self._wrap_key(key)] = value

    def __delitem__(self, key):
        del self._storage[self._wrap_key(key)]

    def __len__(self):
        return len(self._storage)

    def __iter__(self):
        for key in self._storage:
            yield key.unwrapped

    def __repr__(self):
        return 'ObjectIdentityDictionary(%s)' % repr(self._storage)
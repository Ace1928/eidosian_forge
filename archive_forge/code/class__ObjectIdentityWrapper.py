import collections
from typing import Any, Set
import weakref
class _ObjectIdentityWrapper(object):
    """Wraps an object, mapping __eq__ on wrapper to "is" on wrapped.

  Since __eq__ is based on object identity, it's safe to also define __hash__
  based on object ids. This lets us add unhashable types like trackable
  _ListWrapper objects to object-identity collections.
  """
    __slots__ = ['_wrapped', '__weakref__']

    def __init__(self, wrapped):
        self._wrapped = wrapped

    @property
    def unwrapped(self):
        return self._wrapped

    def _assert_type(self, other):
        if not isinstance(other, _ObjectIdentityWrapper):
            raise TypeError('Cannot compare wrapped object with unwrapped object')

    def __lt__(self, other):
        self._assert_type(other)
        return id(self._wrapped) < id(other._wrapped)

    def __gt__(self, other):
        self._assert_type(other)
        return id(self._wrapped) > id(other._wrapped)

    def __eq__(self, other):
        if other is None:
            return False
        self._assert_type(other)
        return self._wrapped is other._wrapped

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return id(self._wrapped)

    def __repr__(self):
        return '<{} wrapping {!r}>'.format(type(self).__name__, self._wrapped)
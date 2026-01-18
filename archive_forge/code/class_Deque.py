import abc
import collections
import collections.abc
import operator
import sys
import typing
class Deque(collections.deque, typing.MutableSequence[T], metaclass=_ExtensionsGenericMeta, extra=collections.deque):
    __slots__ = ()

    def __new__(cls, *args, **kwds):
        if cls._gorg is Deque:
            return collections.deque(*args, **kwds)
        return typing._generic_new(collections.deque, cls, *args, **kwds)
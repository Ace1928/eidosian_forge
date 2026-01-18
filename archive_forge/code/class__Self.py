import abc
import collections
import collections.abc
import operator
import sys
import typing
class _Self(typing._FinalTypingBase, _root=True):
    """Used to spell the type of "self" in classes.

        Example::

          from typing import Self

          class ReturnsSelf:
              def parse(self, data: bytes) -> Self:
                  ...
                  return self

        """
    __slots__ = ()

    def __instancecheck__(self, obj):
        raise TypeError(f'{self} cannot be used with isinstance().')

    def __subclasscheck__(self, cls):
        raise TypeError(f'{self} cannot be used with issubclass().')
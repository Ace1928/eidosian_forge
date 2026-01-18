from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
class LazyWrapper(object):
    """Wrapper for lazily instantiated objects."""

    def __init__(self, func):
        """The init method for LazyWrapper.

    Args:
      func: A function (lambda or otherwise) to lazily evaluate.
    """
        self._func = func

    def __int__(self):
        try:
            return int(self._value)
        except AttributeError:
            self._value = self._func()
            return int(self._value)

    def __eq__(self, other):
        try:
            return self._value == other
        except AttributeError:
            self._value = self._func()
            return self._value == other

    def __repr__(self):
        try:
            return str(self._value)
        except AttributeError:
            self._value = self._func()
            return str(self._value)

    def __str__(self):
        try:
            return str(self._value)
        except AttributeError:
            self._value = self._func()
            return str(self._value)

    def __call__(self):
        """The call method for a LazyWrapper object."""
        try:
            return self._value
        except AttributeError:
            self._value = self._func()
            return self._value

    def __len__(self):
        """The len method for a LazyWrapper object."""
        try:
            return len(self._value)
        except AttributeError:
            self.__call__()
            return len(self._value)

    def __iter__(self):
        """The iter method for a LazyWrapper object."""
        try:
            return self._value.__iter__()
        except AttributeError:
            self.__call__()
            return self._value.__iter__()
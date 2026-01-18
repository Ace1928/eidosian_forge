import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
class ProviderOf(Generic[T]):
    """Can be used to get a provider of an interface, for example:

    >>> def provide_int():
    ...     print('providing')
    ...     return 123
    >>>
    >>> def configure(binder):
    ...     binder.bind(int, to=provide_int)
    >>>
    >>> injector = Injector(configure)
    >>> provider = injector.get(ProviderOf[int])
    >>> value = provider.get()
    providing
    >>> value
    123
    """

    def __init__(self, injector: Injector, interface: Type[T]):
        self._injector = injector
        self._interface = interface

    def __repr__(self) -> str:
        return '%s(%r, %r)' % (type(self).__name__, self._injector, self._interface)

    def get(self) -> T:
        """Get an implementation for the specified interface."""
        return self._injector.get(self._interface)
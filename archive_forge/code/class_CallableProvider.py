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
class CallableProvider(Provider, Generic[T]):
    """Provides something using a callable.

    The callable is called every time new value is requested from the provider.

    There's no need to explicitly use :func:`inject` or :data:`Inject` with the callable as it's
    assumed that, if the callable has annotated parameters, they're meant to be provided
    automatically. It wouldn't make sense any other way, as there's no mechanism to provide
    parameters to the callable at a later time, so either they'll be injected or there'll be
    a `CallError`.

    ::

        >>> class MyClass:
        ...     def __init__(self, value: int) -> None:
        ...         self.value = value
        ...
        >>> def factory():
        ...     print('providing')
        ...     return MyClass(42)
        ...
        >>> def configure(binder):
        ...     binder.bind(MyClass, to=CallableProvider(factory))
        ...
        >>> injector = Injector(configure)
        >>> injector.get(MyClass) is injector.get(MyClass)
        providing
        providing
        False
    """

    def __init__(self, callable: Callable[..., T]):
        self._callable = callable

    def get(self, injector: 'Injector') -> T:
        return injector.call_with_injection(self._callable)

    def __repr__(self) -> str:
        return '%s(%r)' % (type(self).__name__, self._callable)
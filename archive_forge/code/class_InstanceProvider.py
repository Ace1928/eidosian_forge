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
class InstanceProvider(Provider, Generic[T]):
    """Provide a specific instance.

    ::

        >>> class MyType:
        ...     def __init__(self):
        ...         self.contents = []
        >>> def configure(binder):
        ...     binder.bind(MyType, to=InstanceProvider(MyType()))
        ...
        >>> injector = Injector(configure)
        >>> injector.get(MyType) is injector.get(MyType)
        True
        >>> injector.get(MyType).contents.append('x')
        >>> injector.get(MyType).contents
        ['x']
    """

    def __init__(self, instance: T) -> None:
        self._instance = instance

    def get(self, injector: 'Injector') -> T:
        return self._instance

    def __repr__(self) -> str:
        return '%s(%r)' % (type(self).__name__, self._instance)
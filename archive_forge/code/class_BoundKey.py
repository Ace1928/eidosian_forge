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
class BoundKey(tuple):
    """A BoundKey provides a key to a type with pre-injected arguments.

    >>> class A:
    ...   def __init__(self, a, b):
    ...     self.a = a
    ...     self.b = b
    >>> InjectedA = BoundKey(A, a=InstanceProvider(1), b=InstanceProvider(2))
    >>> injector = Injector()
    >>> a = injector.get(InjectedA)
    >>> a.a, a.b
    (1, 2)
    """

    def __new__(cls, interface: Type[T], **kwargs: Any) -> 'BoundKey':
        kwargs_tuple = tuple(sorted(kwargs.items()))
        return super(BoundKey, cls).__new__(cls, (interface, kwargs_tuple))

    @property
    def interface(self) -> Type[T]:
        return self[0]

    @property
    def kwargs(self) -> Dict[str, Any]:
        return dict(self[1])
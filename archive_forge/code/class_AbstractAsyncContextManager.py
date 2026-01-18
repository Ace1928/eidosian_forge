import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
class AbstractAsyncContextManager(abc.ABC):
    """An abstract base class for asynchronous context managers."""
    __class_getitem__ = classmethod(GenericAlias)

    async def __aenter__(self):
        """Return `self` upon entering the runtime context."""
        return self

    @abc.abstractmethod
    async def __aexit__(self, exc_type, exc_value, traceback):
        """Raise any exception triggered within the runtime context."""
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AbstractAsyncContextManager:
            return _collections_abc._check_methods(C, '__aenter__', '__aexit__')
        return NotImplemented
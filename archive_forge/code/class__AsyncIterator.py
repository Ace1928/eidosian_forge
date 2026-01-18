import asyncio
import contextlib
import io
import inspect
import pprint
import sys
import builtins
import pkgutil
from asyncio import iscoroutinefunction
from types import CodeType, ModuleType, MethodType
from unittest.util import safe_repr
from functools import wraps, partial
from threading import RLock
class _AsyncIterator:
    """
    Wraps an iterator in an asynchronous iterator.
    """

    def __init__(self, iterator):
        self.iterator = iterator
        code_mock = NonCallableMock(spec_set=CodeType)
        code_mock.co_flags = inspect.CO_ITERABLE_COROUTINE
        self.__dict__['__code__'] = code_mock

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            pass
        raise StopAsyncIteration
import asyncio
import contextvars
import inspect
import warnings
from .case import TestCase
def _callAsync(self, func, /, *args, **kwargs):
    assert self._asyncioRunner is not None, 'asyncio runner is not initialized'
    assert inspect.iscoroutinefunction(func), f'{func!r} is not an async function'
    return self._asyncioRunner.run(func(*args, **kwargs), context=self._asyncioTestContext)
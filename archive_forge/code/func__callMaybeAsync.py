import asyncio
import contextvars
import inspect
import warnings
from .case import TestCase
def _callMaybeAsync(self, func, /, *args, **kwargs):
    assert self._asyncioRunner is not None, 'asyncio runner is not initialized'
    if inspect.iscoroutinefunction(func):
        return self._asyncioRunner.run(func(*args, **kwargs), context=self._asyncioTestContext)
    else:
        return self._asyncioTestContext.run(func, *args, **kwargs)
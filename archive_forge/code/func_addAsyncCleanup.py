import asyncio
import contextvars
import inspect
import warnings
from .case import TestCase
def addAsyncCleanup(self, func, /, *args, **kwargs):
    self.addCleanup(*(func, *args), **kwargs)
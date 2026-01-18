import asyncio
import contextvars
import inspect
import warnings
from .case import TestCase
def _tearDownAsyncioRunner(self):
    runner = self._asyncioRunner
    runner.close()
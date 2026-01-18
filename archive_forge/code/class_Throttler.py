import functools
import time
import inspect
import collections
import types
import itertools
import warnings
import setuptools.extern.more_itertools
from typing import Callable, TypeVar
class Throttler:
    """
    Rate-limit a function (or other callable)
    """

    def __init__(self, func, max_rate=float('Inf')):
        if isinstance(func, Throttler):
            func = func.func
        self.func = func
        self.max_rate = max_rate
        self.reset()

    def reset(self):
        self.last_called = 0

    def __call__(self, *args, **kwargs):
        self._wait()
        return self.func(*args, **kwargs)

    def _wait(self):
        """ensure at least 1/max_rate seconds from last call"""
        elapsed = time.time() - self.last_called
        must_wait = 1 / self.max_rate - elapsed
        time.sleep(max(0, must_wait))
        self.last_called = time.time()

    def __get__(self, obj, type=None):
        return first_invoke(self._wait, functools.partial(self.func, obj))
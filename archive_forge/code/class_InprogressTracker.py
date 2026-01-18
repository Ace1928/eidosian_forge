from timeit import default_timer
from types import TracebackType
from typing import (
from .decorator import decorate
class InprogressTracker:

    def __init__(self, gauge):
        self._gauge = gauge

    def __enter__(self):
        self._gauge.inc()

    def __exit__(self, typ, value, traceback):
        self._gauge.dec()

    def __call__(self, f: 'F') -> 'F':

        def wrapped(func, *args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate(f, wrapped)
import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
class ANextIter:

    def __init__(self, it, first_fn, *first_args):
        self._it = it
        self._first_fn = first_fn
        self._first_args = first_args

    def __await__(self):
        return self

    def __next__(self):
        if self._first_fn is not None:
            first_fn = self._first_fn
            first_args = self._first_args
            self._first_fn = self._first_args = None
            return self._invoke(first_fn, *first_args)
        else:
            return self._invoke(self._it.__next__)

    def send(self, value):
        return self._invoke(self._it.send, value)

    def throw(self, type, value=None, traceback=None):
        return self._invoke(self._it.throw, type, value, traceback)

    def _invoke(self, fn, *args):
        try:
            result = fn(*args)
        except StopIteration as e:
            raise StopAsyncIteration(e.value)
        except StopAsyncIteration as e:
            raise RuntimeError('async_generator raise StopAsyncIteration') from e
        if _is_wrapped(result):
            raise StopIteration(_unwrap(result))
        else:
            return result
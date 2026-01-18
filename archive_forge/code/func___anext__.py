import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
def __anext__(self):
    return self._do_it(self._it.__next__)
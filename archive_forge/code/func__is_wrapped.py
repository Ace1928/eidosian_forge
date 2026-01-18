import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
def _is_wrapped(box):
    return isinstance(box, YieldWrapper)
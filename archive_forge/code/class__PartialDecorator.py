import sys
from functools import partial
from inspect import isclass
from threading import Lock, RLock
from .arguments import formatargspec
from .__wrapt__ import (FunctionWrapper, BoundFunctionWrapper, ObjectProxy,
class _PartialDecorator(CallableObjectProxy):

    def __enter__(self):
        lock.acquire()
        return lock

    def __exit__(self, *args):
        lock.release()
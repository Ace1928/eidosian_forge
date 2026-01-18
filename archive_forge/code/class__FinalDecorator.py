import sys
from functools import partial
from inspect import isclass
from threading import Lock, RLock
from .arguments import formatargspec
from .__wrapt__ import (FunctionWrapper, BoundFunctionWrapper, ObjectProxy,
class _FinalDecorator(FunctionWrapper):

    def __enter__(self):
        self._self_lock = _synchronized_lock(self.__wrapped__)
        self._self_lock.acquire()
        return self._self_lock

    def __exit__(self, *args):
        self._self_lock.release()
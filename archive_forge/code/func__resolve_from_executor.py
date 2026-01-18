from collections import namedtuple
from functools import partial, wraps
from sys import version_info, exc_info
from threading import RLock
from types import TracebackType
from weakref import WeakKeyDictionary
from .async_ import Async
from .compat import (
from .utils import deprecated, integer_types, string_types, text_type, binary_type, warn
from .promise_list import PromiseList
from .schedulers.immediate import ImmediateScheduler
from typing import TypeVar, Generic
def _resolve_from_executor(self, executor):
    synchronous = True

    def resolve(value):
        self._resolve_callback(value)

    def reject(reason, traceback=None):
        self._reject_callback(reason, synchronous, traceback)
    error = None
    traceback = None
    try:
        executor(resolve, reject)
    except Exception as e:
        traceback = exc_info()[2]
        error = e
    synchronous = False
    if error is not None:
        self._reject_callback(error, True, traceback)
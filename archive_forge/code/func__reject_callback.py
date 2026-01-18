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
def _reject_callback(self, reason, synchronous=False, traceback=None):
    assert isinstance(reason, Exception), 'A promise was rejected with a non-error: {}'.format(reason)
    self._reject(reason, traceback)
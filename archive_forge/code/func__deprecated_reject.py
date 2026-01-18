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
@deprecated('Rejecting directly in a Promise instance is deprecated, as Promise.reject() is now a class method. Please use promise.do_reject() instead.', name='reject')
def _deprecated_reject(self, e):
    self.do_reject(e)
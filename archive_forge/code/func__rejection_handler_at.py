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
def _rejection_handler_at(self, index):
    assert not self._is_following
    assert index > 0
    return self._handlers.get(index * CALLBACK_SIZE - CALLBACK_SIZE + CALLBACK_REJECT_OFFSET)
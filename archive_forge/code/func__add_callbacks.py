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
def _add_callbacks(self, fulfill, reject, promise):
    assert not self._is_following
    if self._handlers is None:
        self._handlers = {}
    index = self._length
    if index > MAX_LENGTH - CALLBACK_SIZE:
        index = 0
        self._length = 0
    if index == 0:
        assert not self._promise0
        assert not self._fulfillment_handler0
        assert not self._rejection_handler0
        self._promise0 = promise
        if callable(fulfill):
            self._fulfillment_handler0 = fulfill
        if callable(reject):
            self._rejection_handler0 = reject
    else:
        base = index * CALLBACK_SIZE - CALLBACK_SIZE
        assert base + CALLBACK_PROMISE_OFFSET not in self._handlers
        assert base + CALLBACK_FULFILL_OFFSET not in self._handlers
        assert base + CALLBACK_REJECT_OFFSET not in self._handlers
        self._handlers[base + CALLBACK_PROMISE_OFFSET] = promise
        if callable(fulfill):
            self._handlers[base + CALLBACK_FULFILL_OFFSET] = fulfill
        if callable(reject):
            self._handlers[base + CALLBACK_REJECT_OFFSET] = reject
    self._length = index + 1
    return index
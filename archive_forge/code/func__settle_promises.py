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
def _settle_promises(self):
    length = self._length
    if length > 0:
        if self._state == STATE_REJECTED:
            reason = self._fulfillment_handler0
            traceback = self._traceback
            self._settle_promise0(self._rejection_handler0, reason, traceback)
            self._reject_promises(length, reason)
        else:
            value = self._rejection_handler0
            self._settle_promise0(self._fulfillment_handler0, value, None)
            self._fulfill_promises(length, value)
        self._length = 0
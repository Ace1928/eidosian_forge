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
def _then(self, did_fulfill=None, did_reject=None):
    promise = self.__class__()
    target = self._target()
    state = target._state
    if state == STATE_PENDING:
        target._add_callbacks(did_fulfill, did_reject, promise)
    else:
        traceback = None
        if state == STATE_FULFILLED:
            value = target._rejection_handler0
            handler = did_fulfill
        elif state == STATE_REJECTED:
            value = target._fulfillment_handler0
            traceback = target._traceback
            handler = did_reject
        async_instance.invoke(partial(target._settle_promise, promise, handler, value, traceback), promise.scheduler)
    return promise
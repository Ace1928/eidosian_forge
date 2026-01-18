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
def _settle_promise_from_handler(self, handler, value, promise):
    value, error_with_tb = try_catch(handler, value)
    if error_with_tb:
        error, tb = error_with_tb
        promise._reject_callback(error, False, tb)
    else:
        promise._resolve_callback(value)
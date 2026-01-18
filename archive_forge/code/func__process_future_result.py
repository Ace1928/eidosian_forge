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
def _process_future_result(resolve, reject):

    def handle_future_result(future):
        try:
            resolve(future.result())
        except Exception as e:
            tb = exc_info()[2]
            reject(e, tb)
    return handle_future_result
from __future__ import annotations
import contextlib
import contextvars
import inspect
import queue as stdlib_queue
import threading
from itertools import count
from typing import TYPE_CHECKING, Generic, TypeVar, overload
import attrs
import outcome
from attrs import define
from sniffio import current_async_library_cvar
import trio
from ._core import (
from ._deprecate import warn_deprecated
from ._sync import CapacityLimiter, Event
from ._util import coroutine_or_error
@contextlib.contextmanager
def _track_active_thread() -> Generator[None, None, None]:
    try:
        active_threads_local = _active_threads_local.get()
    except LookupError:
        active_threads_local = _ActiveThreadCount(0, Event())
        _active_threads_local.set(active_threads_local)
    active_threads_local.count += 1
    try:
        yield
    finally:
        active_threads_local.count -= 1
        if active_threads_local.count == 0:
            active_threads_local.event.set()
            active_threads_local.event = Event()
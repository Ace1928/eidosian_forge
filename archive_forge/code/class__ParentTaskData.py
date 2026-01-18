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
class _ParentTaskData(threading.local):
    """Global due to Threading API, thread local storage for data related to the
    parent task of native Trio threads."""
    token: TrioToken
    abandon_on_cancel: bool
    cancel_register: list[RaiseCancelT | None]
    task_register: list[trio.lowlevel.Task | None]
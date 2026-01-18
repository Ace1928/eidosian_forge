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
@attrs.frozen(eq=False, hash=False, slots=False)
class ThreadPlaceholder:
    name: str
from __future__ import annotations
import sys
import threading
from collections.abc import Awaitable, Callable, Generator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, contextmanager
from inspect import isawaitable
from types import TracebackType
from typing import (
from ._core import _eventloop
from ._core._eventloop import get_async_backend, get_cancelled_exc_class, threadlocals
from ._core._synchronization import Event
from ._core._tasks import CancelScope, create_task_group
from .abc import AsyncBackend
from .abc._tasks import TaskStatus
class _BlockingPortalTaskStatus(TaskStatus):

    def __init__(self, future: Future):
        self._future = future

    def started(self, value: object=None) -> None:
        self._future.set_result(value)
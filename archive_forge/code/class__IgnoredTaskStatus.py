from __future__ import annotations
import math
from collections.abc import Generator
from contextlib import contextmanager
from types import TracebackType
from ..abc._tasks import TaskGroup, TaskStatus
from ._eventloop import get_async_backend
class _IgnoredTaskStatus(TaskStatus[object]):

    def started(self, value: object=None) -> None:
        pass
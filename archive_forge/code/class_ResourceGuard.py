from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass
from types import TracebackType
from sniffio import AsyncLibraryNotFoundError
from ..lowlevel import cancel_shielded_checkpoint, checkpoint, checkpoint_if_cancelled
from ._eventloop import get_async_backend
from ._exceptions import BusyResourceError, WouldBlock
from ._tasks import CancelScope
from ._testing import TaskInfo, get_current_task
class ResourceGuard:
    """
    A context manager for ensuring that a resource is only used by a single task at a
    time.

    Entering this context manager while the previous has not exited it yet will trigger
    :exc:`BusyResourceError`.

    :param action: the action to guard against (visible in the :exc:`BusyResourceError`
        when triggered, e.g. "Another task is already {action} this resource")

    .. versionadded:: 4.1
    """
    __slots__ = ('action', '_guarded')

    def __init__(self, action: str='using'):
        self.action: str = action
        self._guarded = False

    def __enter__(self) -> None:
        if self._guarded:
            raise BusyResourceError(self.action)
        self._guarded = True

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> bool | None:
        self._guarded = False
        return None
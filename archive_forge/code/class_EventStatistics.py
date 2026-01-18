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
@dataclass(frozen=True)
class EventStatistics:
    """
    :ivar int tasks_waiting: number of tasks waiting on :meth:`~.Event.wait`
    """
    tasks_waiting: int
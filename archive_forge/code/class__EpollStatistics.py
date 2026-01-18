from __future__ import annotations
import contextlib
import select
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Literal
import attrs
from .. import _core
from ._io_common import wake_all
from ._run import Task, _public
from ._wakeup_socketpair import WakeupSocketpair
@attrs.frozen(eq=False)
class _EpollStatistics:
    tasks_waiting_read: int
    tasks_waiting_write: int
    backend: Literal['epoll'] = attrs.field(init=False, default='epoll')
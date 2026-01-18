from __future__ import annotations
import weakref
from typing import TYPE_CHECKING, Callable, Union
import pytest
from .. import _core
from .._sync import *
from .._timeouts import sleep_forever
from ..testing import assert_checkpoints, wait_all_tasks_blocked
from .._channel import open_memory_channel
from .._sync import AsyncContextManagerMixin
class ChannelLock1(AsyncContextManagerMixin):

    def __init__(self, capacity: int) -> None:
        self.s, self.r = open_memory_channel[None](capacity)
        for _ in range(capacity - 1):
            self.s.send_nowait(None)

    def acquire_nowait(self) -> None:
        self.s.send_nowait(None)

    async def acquire(self) -> None:
        await self.s.send(None)

    def release(self) -> None:
        self.r.receive_nowait()
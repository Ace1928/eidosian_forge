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
class CapacityLimiterAdapter(CapacityLimiter):
    _internal_limiter: CapacityLimiter | None = None

    def __new__(cls, total_tokens: float) -> CapacityLimiterAdapter:
        return object.__new__(cls)

    def __init__(self, total_tokens: float) -> None:
        self.total_tokens = total_tokens

    @property
    def _limiter(self) -> CapacityLimiter:
        if self._internal_limiter is None:
            self._internal_limiter = get_async_backend().create_capacity_limiter(self._total_tokens)
        return self._internal_limiter

    async def __aenter__(self) -> None:
        await self._limiter.__aenter__()

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> bool | None:
        return await self._limiter.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def total_tokens(self) -> float:
        if self._internal_limiter is None:
            return self._total_tokens
        return self._internal_limiter.total_tokens

    @total_tokens.setter
    def total_tokens(self, value: float) -> None:
        if not isinstance(value, int) and value is not math.inf:
            raise TypeError('total_tokens must be an int or math.inf')
        elif value < 1:
            raise ValueError('total_tokens must be >= 1')
        if self._internal_limiter is None:
            self._total_tokens = value
            return
        self._limiter.total_tokens = value

    @property
    def borrowed_tokens(self) -> int:
        if self._internal_limiter is None:
            return 0
        return self._internal_limiter.borrowed_tokens

    @property
    def available_tokens(self) -> float:
        if self._internal_limiter is None:
            return self._total_tokens
        return self._internal_limiter.available_tokens

    def acquire_nowait(self) -> None:
        self._limiter.acquire_nowait()

    def acquire_on_behalf_of_nowait(self, borrower: object) -> None:
        self._limiter.acquire_on_behalf_of_nowait(borrower)

    async def acquire(self) -> None:
        await self._limiter.acquire()

    async def acquire_on_behalf_of(self, borrower: object) -> None:
        await self._limiter.acquire_on_behalf_of(borrower)

    def release(self) -> None:
        self._limiter.release()

    def release_on_behalf_of(self, borrower: object) -> None:
        self._limiter.release_on_behalf_of(borrower)

    def statistics(self) -> CapacityLimiterStatistics:
        if self._internal_limiter is None:
            return CapacityLimiterStatistics(borrowed_tokens=0, total_tokens=self.total_tokens, borrowers=(), tasks_waiting=0)
        return self._internal_limiter.statistics()
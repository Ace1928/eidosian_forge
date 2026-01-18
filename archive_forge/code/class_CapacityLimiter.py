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
class CapacityLimiter:

    def __new__(cls, total_tokens: float) -> CapacityLimiter:
        try:
            return get_async_backend().create_capacity_limiter(total_tokens)
        except AsyncLibraryNotFoundError:
            return CapacityLimiterAdapter(total_tokens)

    async def __aenter__(self) -> None:
        raise NotImplementedError

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> bool | None:
        raise NotImplementedError

    @property
    def total_tokens(self) -> float:
        """
        The total number of tokens available for borrowing.

        This is a read-write property. If the total number of tokens is increased, the
        proportionate number of tasks waiting on this limiter will be granted their
        tokens.

        .. versionchanged:: 3.0
            The property is now writable.

        """
        raise NotImplementedError

    @total_tokens.setter
    def total_tokens(self, value: float) -> None:
        raise NotImplementedError

    @property
    def borrowed_tokens(self) -> int:
        """The number of tokens that have currently been borrowed."""
        raise NotImplementedError

    @property
    def available_tokens(self) -> float:
        """The number of tokens currently available to be borrowed"""
        raise NotImplementedError

    def acquire_nowait(self) -> None:
        """
        Acquire a token for the current task without waiting for one to become
        available.

        :raises ~anyio.WouldBlock: if there are no tokens available for borrowing

        """
        raise NotImplementedError

    def acquire_on_behalf_of_nowait(self, borrower: object) -> None:
        """
        Acquire a token without waiting for one to become available.

        :param borrower: the entity borrowing a token
        :raises ~anyio.WouldBlock: if there are no tokens available for borrowing

        """
        raise NotImplementedError

    async def acquire(self) -> None:
        """
        Acquire a token for the current task, waiting if necessary for one to become
        available.

        """
        raise NotImplementedError

    async def acquire_on_behalf_of(self, borrower: object) -> None:
        """
        Acquire a token, waiting if necessary for one to become available.

        :param borrower: the entity borrowing a token

        """
        raise NotImplementedError

    def release(self) -> None:
        """
        Release the token held by the current task.

        :raises RuntimeError: if the current task has not borrowed a token from this
            limiter.

        """
        raise NotImplementedError

    def release_on_behalf_of(self, borrower: object) -> None:
        """
        Release the token held by the given borrower.

        :raises RuntimeError: if the borrower has not borrowed a token from this
            limiter.

        """
        raise NotImplementedError

    def statistics(self) -> CapacityLimiterStatistics:
        """
        Return statistics about the current state of this limiter.

        .. versionadded:: 3.0

        """
        raise NotImplementedError
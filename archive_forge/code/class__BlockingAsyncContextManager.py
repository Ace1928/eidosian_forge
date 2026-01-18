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
class _BlockingAsyncContextManager(Generic[T_co], AbstractContextManager):
    _enter_future: Future[T_co]
    _exit_future: Future[bool | None]
    _exit_event: Event
    _exit_exc_info: tuple[type[BaseException] | None, BaseException | None, TracebackType | None] = (None, None, None)

    def __init__(self, async_cm: AsyncContextManager[T_co], portal: BlockingPortal):
        self._async_cm = async_cm
        self._portal = portal

    async def run_async_cm(self) -> bool | None:
        try:
            self._exit_event = Event()
            value = await self._async_cm.__aenter__()
        except BaseException as exc:
            self._enter_future.set_exception(exc)
            raise
        else:
            self._enter_future.set_result(value)
        try:
            await self._exit_event.wait()
        finally:
            result = await self._async_cm.__aexit__(*self._exit_exc_info)
            return result

    def __enter__(self) -> T_co:
        self._enter_future = Future()
        self._exit_future = self._portal.start_task_soon(self.run_async_cm)
        return self._enter_future.result()

    def __exit__(self, __exc_type: type[BaseException] | None, __exc_value: BaseException | None, __traceback: TracebackType | None) -> bool | None:
        self._exit_exc_info = (__exc_type, __exc_value, __traceback)
        self._portal.call(self._exit_event.set)
        return self._exit_future.result()
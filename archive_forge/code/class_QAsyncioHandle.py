from PySide6.QtCore import (QCoreApplication, QDateTime, QDeadlineTimer,
from . import futures
from . import tasks
import asyncio
import collections.abc
import concurrent.futures
import contextvars
import enum
import os
import signal
import socket
import subprocess
import typing
import warnings
class QAsyncioHandle:

    class HandleState(enum.Enum):
        PENDING = enum.auto()
        CANCELLED = enum.auto()
        DONE = enum.auto()

    def __init__(self, callback: typing.Callable, args: typing.Tuple, loop: QAsyncioEventLoop, context: typing.Optional[contextvars.Context], is_threadsafe: typing.Optional[bool]=False) -> None:
        self._callback = callback
        self._args = args
        self._loop = loop
        self._context = context
        self._is_threadsafe = is_threadsafe
        self._timeout = 0
        self._state = QAsyncioHandle.HandleState.PENDING
        self._start()

    def _schedule_event(self, timeout: int, func: typing.Callable) -> None:
        if not self._loop.is_closed() and (not self._loop._quit_from_outside):
            if self._is_threadsafe:
                QTimer.singleShot(timeout, self._loop, func)
            else:
                QTimer.singleShot(timeout, func)

    def _start(self) -> None:
        self._schedule_event(self._timeout, lambda: self._cb())

    @Slot()
    def _cb(self) -> None:
        if self._state == QAsyncioHandle.HandleState.PENDING:
            if self._context is not None:
                self._context.run(self._callback, *self._args)
            else:
                self._callback(*self._args)
            self._state = QAsyncioHandle.HandleState.DONE

    def cancel(self) -> None:
        if self._state == QAsyncioHandle.HandleState.PENDING:
            self._state = QAsyncioHandle.HandleState.CANCELLED

    def cancelled(self) -> bool:
        return self._state == QAsyncioHandle.HandleState.CANCELLED
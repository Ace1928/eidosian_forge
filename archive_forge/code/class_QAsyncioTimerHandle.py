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
class QAsyncioTimerHandle(QAsyncioHandle, asyncio.TimerHandle):

    def __init__(self, when: float, callback: typing.Callable, args: typing.Tuple, loop: QAsyncioEventLoop, context: typing.Optional[contextvars.Context], is_threadsafe: typing.Optional[bool]=False) -> None:
        QAsyncioHandle.__init__(self, callback, args, loop, context, is_threadsafe)
        self._when = when
        self._timeout = int(max(self._when - self._loop.time(), 0) * 1000)
        QAsyncioHandle._start(self)

    def _start(self) -> None:
        pass

    def when(self) -> float:
        return self._when
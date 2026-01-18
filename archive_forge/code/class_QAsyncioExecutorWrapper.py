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
class QAsyncioExecutorWrapper(QObject):

    def __init__(self, func: typing.Callable, *args: typing.Tuple) -> None:
        super().__init__()
        self._loop: QEventLoop
        self._func = func
        self._args = args
        self._result = None
        self._exception = None

    def _cb(self):
        try:
            self._result = self._func(*self._args)
        except BaseException as e:
            self._exception = e
        self._loop.exit()

    def do(self):
        self._loop = QEventLoop()
        asyncio.events._set_running_loop(self._loop)
        QTimer.singleShot(0, self._loop, lambda: self._cb())
        self._loop.exec()
        if self._exception is not None:
            raise self._exception
        return self._result

    def exit(self):
        self._loop.exit()
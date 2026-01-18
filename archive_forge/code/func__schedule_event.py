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
def _schedule_event(self, timeout: int, func: typing.Callable) -> None:
    if not self._loop.is_closed() and (not self._loop._quit_from_outside):
        if self._is_threadsafe:
            QTimer.singleShot(timeout, self._loop, func)
        else:
            QTimer.singleShot(timeout, func)
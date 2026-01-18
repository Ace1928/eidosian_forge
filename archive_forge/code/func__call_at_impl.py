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
def _call_at_impl(self, when: typing.Union[int, float], callback: typing.Callable, *args: typing.Any, context: typing.Optional[contextvars.Context]=None, is_threadsafe: typing.Optional[bool]=False) -> asyncio.TimerHandle:
    if not isinstance(when, (int, float)):
        raise TypeError('when must be an int or float')
    if self.is_closed():
        raise RuntimeError('Event loop is closed')
    return QAsyncioTimerHandle(when, callback, args, self, context, is_threadsafe=is_threadsafe)
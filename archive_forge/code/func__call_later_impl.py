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
def _call_later_impl(self, delay: typing.Union[int, float], callback: typing.Callable, *args: typing.Any, context: typing.Optional[contextvars.Context]=None, is_threadsafe: typing.Optional[bool]=False) -> asyncio.TimerHandle:
    if not isinstance(delay, (int, float)):
        raise TypeError('delay must be an int or float')
    return self._call_at_impl(self.time() + delay, callback, *args, context=context, is_threadsafe=is_threadsafe)
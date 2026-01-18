import asyncio
import atexit
import concurrent.futures
import errno
import functools
import select
import socket
import sys
import threading
import typing
import warnings
from tornado.gen import convert_yielded
from tornado.ioloop import IOLoop, _Selectable
from typing import (
def add_callback_from_signal(self, callback: Callable, *args: Any, **kwargs: Any) -> None:
    warnings.warn('add_callback_from_signal is deprecated', DeprecationWarning)
    try:
        self.asyncio_loop.call_soon_threadsafe(self._run_callback, functools.partial(callback, *args, **kwargs))
    except RuntimeError:
        pass
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
def _run_select(self) -> None:
    while True:
        with self._select_cond:
            while self._select_args is None and (not self._closing_selector):
                self._select_cond.wait()
            if self._closing_selector:
                return
            assert self._select_args is not None
            to_read, to_write = self._select_args
            self._select_args = None
        try:
            rs, ws, xs = select.select(to_read, to_write, to_write)
            ws = ws + xs
        except OSError as e:
            if e.errno == getattr(errno, 'WSAENOTSOCK', errno.EBADF):
                rs, _, _ = select.select([self._waker_r.fileno()], [], [], 0)
                if rs:
                    ws = []
                else:
                    raise
            else:
                raise
        try:
            self._real_loop.call_soon_threadsafe(self._handle_select, rs, ws)
        except RuntimeError:
            pass
        except AttributeError:
            pass
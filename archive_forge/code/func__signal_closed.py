import asyncio
import collections
import errno
import io
import numbers
import os
import socket
import ssl
import sys
import re
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import ioloop
from tornado.log import gen_log
from tornado.netutil import ssl_wrap_socket, _client_ssl_defaults, _server_ssl_defaults
from tornado.util import errno_from_exception
import typing
from typing import (
from types import TracebackType
def _signal_closed(self) -> None:
    futures = []
    if self._read_future is not None:
        futures.append(self._read_future)
        self._read_future = None
    futures += [future for _, future in self._write_futures]
    self._write_futures.clear()
    if self._connect_future is not None:
        futures.append(self._connect_future)
        self._connect_future = None
    for future in futures:
        if not future.done():
            future.set_exception(StreamClosedError(real_error=self.error))
        try:
            future.exception()
        except asyncio.CancelledError:
            pass
    if self._ssl_connect_future is not None:
        if not self._ssl_connect_future.done():
            if self.error is not None:
                self._ssl_connect_future.set_exception(self.error)
            else:
                self._ssl_connect_future.set_exception(StreamClosedError())
        self._ssl_connect_future.exception()
        self._ssl_connect_future = None
    if self._close_callback is not None:
        cb = self._close_callback
        self._close_callback = None
        self.io_loop.add_callback(cb)
    self._write_buffer = None
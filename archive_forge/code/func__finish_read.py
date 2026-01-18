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
def _finish_read(self, size: int) -> None:
    if self._user_read_buffer:
        self._read_buffer = self._after_user_read_buffer or bytearray()
        self._after_user_read_buffer = None
        self._read_buffer_size = len(self._read_buffer)
        self._user_read_buffer = False
        result = size
    else:
        result = self._consume(size)
    if self._read_future is not None:
        future = self._read_future
        self._read_future = None
        future_set_result_unless_cancelled(future, result)
    self._maybe_add_error_listener()
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
def _try_inline_read(self) -> None:
    """Attempt to complete the current read operation from buffered data.

        If the read can be completed without blocking, schedules the
        read callback on the next IOLoop iteration; otherwise starts
        listening for reads on the socket.
        """
    pos = self._find_read_pos()
    if pos is not None:
        self._read_from_buffer(pos)
        return
    self._check_closed()
    pos = self._read_to_buffer_loop()
    if pos is not None:
        self._read_from_buffer(pos)
        return
    if not self.closed():
        self._add_io_state(ioloop.IOLoop.READ)
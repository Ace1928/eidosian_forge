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
def _maybe_add_error_listener(self) -> None:
    if self._state is None or self._state == ioloop.IOLoop.ERROR:
        if not self.closed() and self._read_buffer_size == 0 and (self._close_callback is not None):
            self._add_io_state(ioloop.IOLoop.READ)
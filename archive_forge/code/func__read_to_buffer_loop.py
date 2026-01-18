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
def _read_to_buffer_loop(self) -> Optional[int]:
    if self._read_bytes is not None:
        target_bytes = self._read_bytes
    elif self._read_max_bytes is not None:
        target_bytes = self._read_max_bytes
    elif self.reading():
        target_bytes = None
    else:
        target_bytes = 0
    next_find_pos = 0
    while not self.closed():
        if self._read_to_buffer() == 0:
            break
        if target_bytes is not None and self._read_buffer_size >= target_bytes:
            break
        if self._read_buffer_size >= next_find_pos:
            pos = self._find_read_pos()
            if pos is not None:
                return pos
            next_find_pos = self._read_buffer_size * 2
    return self._find_read_pos()
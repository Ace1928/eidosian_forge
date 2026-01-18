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
def _find_read_pos(self) -> Optional[int]:
    """Attempts to find a position in the read buffer that satisfies
        the currently-pending read.

        Returns a position in the buffer if the current read can be satisfied,
        or None if it cannot.
        """
    if self._read_bytes is not None and (self._read_buffer_size >= self._read_bytes or (self._read_partial and self._read_buffer_size > 0)):
        num_bytes = min(self._read_bytes, self._read_buffer_size)
        return num_bytes
    elif self._read_delimiter is not None:
        if self._read_buffer:
            loc = self._read_buffer.find(self._read_delimiter)
            if loc != -1:
                delimiter_len = len(self._read_delimiter)
                self._check_max_bytes(self._read_delimiter, loc + delimiter_len)
                return loc + delimiter_len
            self._check_max_bytes(self._read_delimiter, self._read_buffer_size)
    elif self._read_regex is not None:
        if self._read_buffer:
            m = self._read_regex.search(self._read_buffer)
            if m is not None:
                loc = m.end()
                self._check_max_bytes(self._read_regex, loc)
                return loc
            self._check_max_bytes(self._read_regex, self._read_buffer_size)
    return None
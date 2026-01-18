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
def _handle_connect(self) -> None:
    super()._handle_connect()
    if self.closed():
        return
    self.io_loop.remove_handler(self.socket)
    old_state = self._state
    assert old_state is not None
    self._state = None
    self.socket = ssl_wrap_socket(self.socket, self._ssl_options, server_hostname=self._server_hostname, do_handshake_on_connect=False, server_side=False)
    self._add_io_state(old_state)
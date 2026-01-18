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
class PipeIOStream(BaseIOStream):
    """Pipe-based `IOStream` implementation.

    The constructor takes an integer file descriptor (such as one returned
    by `os.pipe`) rather than an open file object.  Pipes are generally
    one-way, so a `PipeIOStream` can be used for reading or writing but not
    both.

    ``PipeIOStream`` is only available on Unix-based platforms.
    """

    def __init__(self, fd: int, *args: Any, **kwargs: Any) -> None:
        self.fd = fd
        self._fio = io.FileIO(self.fd, 'r+')
        if sys.platform == 'win32':
            raise AssertionError('PipeIOStream is not supported on Windows')
        os.set_blocking(fd, False)
        super().__init__(*args, **kwargs)

    def fileno(self) -> int:
        return self.fd

    def close_fd(self) -> None:
        self._fio.close()

    def write_to_fd(self, data: memoryview) -> int:
        try:
            return os.write(self.fd, data)
        finally:
            del data

    def read_from_fd(self, buf: Union[bytearray, memoryview]) -> Optional[int]:
        try:
            return self._fio.readinto(buf)
        except (IOError, OSError) as e:
            if errno_from_exception(e) == errno.EBADF:
                self.close(exc_info=e)
                return None
            else:
                raise
        finally:
            del buf
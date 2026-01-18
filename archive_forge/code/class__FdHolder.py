from __future__ import annotations
import errno
import os
import sys
from typing import TYPE_CHECKING
import trio
from ._abc import Stream
from ._util import ConflictDetector, final
class _FdHolder:
    fd: int

    def __init__(self, fd: int) -> None:
        self.fd = -1
        if not isinstance(fd, int):
            raise TypeError('file descriptor must be an int')
        self.fd = fd
        self._original_is_blocking = os.get_blocking(fd)
        os.set_blocking(fd, False)

    @property
    def closed(self) -> bool:
        return self.fd == -1

    def _raw_close(self) -> None:
        if self.closed:
            return
        fd = self.fd
        self.fd = -1
        os.set_blocking(fd, self._original_is_blocking)
        os.close(fd)

    def __del__(self) -> None:
        self._raw_close()

    def close(self) -> None:
        if not self.closed:
            trio.lowlevel.notify_closing(self.fd)
            self._raw_close()
import asyncio
import functools
import pycares
import socket
import sys
from typing import (
from . import error
def _timer_cb(self) -> None:
    if self._read_fds or self._write_fds:
        self._channel.process_fd(pycares.ARES_SOCKET_BAD, pycares.ARES_SOCKET_BAD)
        self._start_timer()
    else:
        self._timer = None
from __future__ import annotations
from collections import deque
from functools import partial
from io import BytesIO
from time import time
from kombu.asynchronous.hub import READ, WRITE, Hub, get_event_loop
from kombu.exceptions import HttpError
from kombu.utils.encoding import bytes_to_str
from .base import BaseClient
def _handle_socket(self, event, fd, multi, data, _pycurl=pycurl):
    if event == _pycurl.POLL_REMOVE:
        if fd in self._fds:
            self._fds.pop(fd, None)
    elif event == _pycurl.POLL_IN:
        self._fds[fd] = READ
    elif event == _pycurl.POLL_OUT:
        self._fds[fd] = WRITE
    elif event == _pycurl.POLL_INOUT:
        self._fds[fd] = READ | WRITE
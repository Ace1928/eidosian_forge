from __future__ import annotations
from collections import deque
from functools import partial
from io import BytesIO
from time import time
from kombu.asynchronous.hub import READ, WRITE, Hub, get_event_loop
from kombu.exceptions import HttpError
from kombu.utils.encoding import bytes_to_str
from .base import BaseClient
def _process_pending_requests(self):
    while 1:
        q, succeeded, failed = self._multi.info_read()
        for curl in succeeded:
            self._process(curl)
        for curl, errno, reason in failed:
            self._process(curl, errno, reason)
        if q == 0:
            break
    self._process_queue()
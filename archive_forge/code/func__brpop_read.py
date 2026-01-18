from __future__ import annotations
import functools
import numbers
import socket
from bisect import bisect
from collections import namedtuple
from contextlib import contextmanager
from queue import Empty
from time import time
from vine import promise
from kombu.exceptions import InconsistencyError, VersionMismatch
from kombu.log import get_logger
from kombu.utils.compat import register_after_fork
from kombu.utils.encoding import bytes_to_str
from kombu.utils.eventio import ERR, READ, poll
from kombu.utils.functional import accepts_argument
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from kombu.utils.scheduling import cycle_by_name
from kombu.utils.url import _parse_url
from . import virtual
def _brpop_read(self, **options):
    try:
        try:
            dest__item = self.client.parse_response(self.client.connection, 'BRPOP', **options)
        except self.connection_errors:
            self.client.connection.disconnect()
            raise
        if dest__item:
            dest, item = dest__item
            dest = bytes_to_str(dest).rsplit(self.sep, 1)[0]
            self._queue_cycle.rotate(dest)
            self.connection._deliver(loads(bytes_to_str(item)), dest)
            return True
        else:
            raise Empty()
    finally:
        self._in_poll = None
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
def _close_clients(self):
    for attr in ('client', 'subclient'):
        try:
            client = self.__dict__[attr]
            connection, client.connection = (client.connection, None)
            connection.disconnect()
        except (KeyError, AttributeError, self.ResponseError):
            pass
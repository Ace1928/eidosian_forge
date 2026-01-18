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
def _brpop_start(self, timeout=1):
    queues = self._queue_cycle.consume(len(self.active_queues))
    if not queues:
        return
    keys = [self._q_for_pri(queue, pri) for pri in self.priority_steps for queue in queues] + [timeout or 0]
    self._in_poll = self.client.connection
    command_args = ['BRPOP', *keys]
    if self.global_keyprefix:
        command_args = self.client._prefix_args(command_args)
    self.client.connection.send_command(*command_args)
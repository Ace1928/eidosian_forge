from __future__ import annotations
import os
import socket
import sys
from contextlib import contextmanager
from itertools import count, cycle
from operator import itemgetter
from typing import TYPE_CHECKING, Any
from kombu import exceptions
from .log import get_logger
from .resource import Resource
from .transport import get_transport_cls, supports_librabbitmq
from .utils.collections import HashedSeq
from .utils.functional import dictfilter, lazy, retry_over_time, shufflecycle
from .utils.objects import cached_property
from .utils.url import as_url, maybe_sanitize_url, parse_url, quote, urlparse
def _extract_failover_opts(self):
    conn_opts = {'timeout': self.connect_timeout}
    transport_opts = self.transport_options
    if transport_opts:
        if 'max_retries' in transport_opts:
            conn_opts['max_retries'] = transport_opts['max_retries']
        if 'interval_start' in transport_opts:
            conn_opts['interval_start'] = transport_opts['interval_start']
        if 'interval_step' in transport_opts:
            conn_opts['interval_step'] = transport_opts['interval_step']
        if 'interval_max' in transport_opts:
            conn_opts['interval_max'] = transport_opts['interval_max']
        if 'connect_retries_timeout' in transport_opts:
            conn_opts['timeout'] = transport_opts['connect_retries_timeout']
    return conn_opts
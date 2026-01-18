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
def _ensured(*args, **kwargs):
    got_connection = 0
    conn_errors = self.recoverable_connection_errors
    chan_errors = self.recoverable_channel_errors
    has_modern_errors = hasattr(self.transport, 'recoverable_connection_errors')
    with self._reraise_as_library_errors():
        for retries in count(0):
            try:
                return fun(*args, **kwargs)
            except retry_errors as exc:
                if max_retries is not None and retries >= max_retries:
                    raise
                self._debug('ensure retry policy error: %r', exc, exc_info=1)
            except conn_errors as exc:
                if got_connection and (not has_modern_errors):
                    raise
                if max_retries is not None and retries >= max_retries:
                    raise
                self._debug('ensure connection error: %r', exc, exc_info=1)
                self.collect()
                errback and errback(exc, 0)
                remaining_retries = None
                if max_retries is not None:
                    remaining_retries = max(max_retries - retries, 1)
                self._ensure_connection(errback, remaining_retries, interval_start, interval_step, interval_max, reraise_as_library_errors=False)
                channel = self.default_channel
                obj.revive(channel)
                if on_revive:
                    on_revive(channel)
                got_connection += 1
            except chan_errors as exc:
                if max_retries is not None and retries > max_retries:
                    raise
                self._debug('ensure channel error: %r', exc, exc_info=1)
                errback and errback(exc, 0)
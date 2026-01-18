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
@contextmanager
def _reraise_as_library_errors(self, ConnectionError=exceptions.OperationalError, ChannelError=exceptions.OperationalError):
    try:
        yield
    except (ConnectionError, ChannelError):
        raise
    except self.recoverable_connection_errors as exc:
        raise ConnectionError(str(exc)) from exc
    except self.recoverable_channel_errors as exc:
        raise ChannelError(str(exc)) from exc
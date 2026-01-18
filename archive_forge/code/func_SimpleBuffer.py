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
def SimpleBuffer(self, name, no_ack=None, queue_opts=None, queue_args=None, exchange_opts=None, channel=None, **kwargs):
    """Simple ephemeral queue API.

        Create new :class:`~kombu.simple.SimpleQueue` using a channel
        from this connection.

        See Also
        --------
            Same as :meth:`SimpleQueue`, but configured with buffering
            semantics. The resulting queue and exchange will not be durable,
            also auto delete is enabled. Messages will be transient (not
            persistent), and acknowledgments are disabled (``no_ack``).
        """
    from .simple import SimpleBuffer
    return SimpleBuffer(channel or self, name, no_ack, queue_opts, queue_args, exchange_opts, **kwargs)
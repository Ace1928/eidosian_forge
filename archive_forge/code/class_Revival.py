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
class Revival:
    __name__ = getattr(fun, '__name__', None)
    __module__ = getattr(fun, '__module__', None)
    __doc__ = getattr(fun, '__doc__', None)

    def __init__(self, connection):
        self.connection = connection

    def revive(self, channel):
        channels[0] = channel

    def __call__(self, *args, **kwargs):
        if channels[0] is None:
            self.revive(self.connection.default_channel)
        return (fun(*args, channel=channels[0], **kwargs), channels[0])
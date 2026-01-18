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
class PrefixedRedisPipeline(GlobalKeyPrefixMixin, redis.client.Pipeline):
    """Custom Redis pipeline that takes global_keyprefix into consideration.

    As the ``PrefixedStrictRedis`` client uses the `global_keyprefix` to prefix
    the keys it uses, the pipeline called by the client must be able to prefix
    the keys as well.
    """

    def __init__(self, *args, **kwargs):
        self.global_keyprefix = kwargs.pop('global_keyprefix', '')
        redis.client.Pipeline.__init__(self, *args, **kwargs)
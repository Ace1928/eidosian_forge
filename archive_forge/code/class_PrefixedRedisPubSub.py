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
class PrefixedRedisPubSub(redis.client.PubSub):
    """Redis pubsub client that takes global_keyprefix into consideration."""
    PUBSUB_COMMANDS = ('SUBSCRIBE', 'UNSUBSCRIBE', 'PSUBSCRIBE', 'PUNSUBSCRIBE')

    def __init__(self, *args, **kwargs):
        self.global_keyprefix = kwargs.pop('global_keyprefix', '')
        super().__init__(*args, **kwargs)

    def _prefix_args(self, args):
        args = list(args)
        command = args.pop(0)
        if command in self.PUBSUB_COMMANDS:
            args = [self.global_keyprefix + str(arg) for arg in args]
        return [command, *args]

    def parse_response(self, *args, **kwargs):
        """Parse a response from the Redis server.

        Method wraps ``PubSub.parse_response()`` to remove prefixes of keys
        returned by redis command.
        """
        ret = super().parse_response(*args, **kwargs)
        if ret is None:
            return ret
        message_type, *channels, message = ret
        return [message_type, *[channel[len(self.global_keyprefix):] for channel in channels], message]

    def execute_command(self, *args, **kwargs):
        return super().execute_command(*self._prefix_args(args), **kwargs)
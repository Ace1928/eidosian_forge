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
class GlobalKeyPrefixMixin:
    """Mixin to provide common logic for global key prefixing.

    Overriding all the methods used by Kombu with the same key prefixing logic
    would be cumbersome and inefficient. Hence, we override the command
    execution logic that is called by all commands.
    """
    PREFIXED_SIMPLE_COMMANDS = ['HDEL', 'HGET', 'HLEN', 'HSET', 'LLEN', 'LPUSH', 'PUBLISH', 'RPUSH', 'RPOP', 'SADD', 'SREM', 'SET', 'SMEMBERS', 'ZADD', 'ZREM', 'ZREVRANGEBYSCORE']
    PREFIXED_COMPLEX_COMMANDS = {'DEL': {'args_start': 0, 'args_end': None}, 'BRPOP': {'args_start': 0, 'args_end': -1}, 'EVALSHA': {'args_start': 2, 'args_end': 3}, 'WATCH': {'args_start': 0, 'args_end': None}}

    def _prefix_args(self, args):
        args = list(args)
        command = args.pop(0)
        if command in self.PREFIXED_SIMPLE_COMMANDS:
            args[0] = self.global_keyprefix + str(args[0])
        elif command in self.PREFIXED_COMPLEX_COMMANDS:
            args_start = self.PREFIXED_COMPLEX_COMMANDS[command]['args_start']
            args_end = self.PREFIXED_COMPLEX_COMMANDS[command]['args_end']
            pre_args = args[:args_start] if args_start > 0 else []
            post_args = []
            if args_end is not None:
                post_args = args[args_end:]
            args = pre_args + [self.global_keyprefix + str(arg) for arg in args[args_start:args_end]] + post_args
        return [command, *args]

    def parse_response(self, connection, command_name, **options):
        """Parse a response from the Redis server.

        Method wraps ``redis.parse_response()`` to remove prefixes of keys
        returned by redis command.
        """
        ret = super().parse_response(connection, command_name, **options)
        if command_name == 'BRPOP' and ret:
            key, value = ret
            key = key[len(self.global_keyprefix):]
            return (key, value)
        return ret

    def execute_command(self, *args, **kwargs):
        return super().execute_command(*self._prefix_args(args), **kwargs)

    def pipeline(self, transaction=True, shard_hint=None):
        return PrefixedRedisPipeline(self.connection_pool, self.response_callbacks, transaction, shard_hint, global_keyprefix=self.global_keyprefix)
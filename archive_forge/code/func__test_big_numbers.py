import asyncio
import functools
import random
import re
import socket
from datetime import timedelta
from unittest import SkipTest, mock
from statsd import StatsClient
from statsd import TCPStatsClient
from statsd import UnixSocketStatsClient
def _test_big_numbers(cl, proto):
    num = 1234568901234
    tests = (('gauge', 'foo:1234568901234|g'), ('incr', 'foo:1234568901234|c'), ('timing', 'foo:1234568901234.000000|ms'))

    def _check(method, result):
        cl._sock.reset_mock()
        getattr(cl, method)('foo', num)
        _sock_check(cl._sock, 1, proto, result)
    for method, result in tests:
        _check(method, result)
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
def _test_timer_decorator_rate(cl, proto):

    @cl.timer('foo', rate=0.1)
    def foo(a, b):
        return [b, a]

    @cl.timer('bar', rate=0.2)
    def bar(a, b=2, c=3):
        return [c, b, a]
    eq_([2, 4], foo(4, 2))
    _timer_check(cl._sock, 1, proto, 'foo', 'ms|@0.1')
    eq_([3, 2, 5], bar(5))
    _timer_check(cl._sock, 2, proto, 'bar', 'ms|@0.2')
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
def _test_gauge_delta(cl, proto):
    tests = ((12, '+12'), (-13, '-13'), (1.2, '+1.2'), (-1.3, '-1.3'))

    def _check(num, result):
        cl._sock.reset_mock()
        cl.gauge('foo', num, delta=True)
        _sock_check(cl._sock, 1, proto, 'foo:%s|g' % result)
    for num, result in tests:
        _check(num, result)
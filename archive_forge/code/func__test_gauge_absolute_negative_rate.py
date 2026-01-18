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
def _test_gauge_absolute_negative_rate(cl, proto, mock_random):
    mock_random.return_value = -1
    cl.gauge('foo', -1, rate=0.5, delta=False)
    _sock_check(cl._sock, 1, proto, 'foo:0|g\nfoo:-1|g')
    mock_random.return_value = 2
    cl.gauge('foo', -2, rate=0.5, delta=False)
    _sock_check(cl._sock, 1, proto, 'foo:0|g\nfoo:-1|g')
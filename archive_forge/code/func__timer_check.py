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
def _timer_check(sock, count, proto, start, end):
    send = send_method[proto](sock)
    eq_(send.call_count, count)
    value = send.call_args[0][0].decode('ascii')
    exp = re.compile('^%s:\\d+|%s$' % (start, end))
    assert exp.match(value)
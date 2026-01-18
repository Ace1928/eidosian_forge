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
def _test_timer_object_no_send(cl, proto):
    t = cl.timer('foo').start()
    t.stop(send=False)
    _sock_check(cl._sock, 0, proto)
    t.send()
    _timer_check(cl._sock, 1, proto, 'foo', 'ms')
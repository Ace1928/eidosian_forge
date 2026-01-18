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
def _test_timer_object_rate(cl, proto):
    t = cl.timer('foo', rate=0.5)
    t.start()
    t.stop()
    _timer_check(cl._sock, 1, proto, 'foo', 'ms@0.5')
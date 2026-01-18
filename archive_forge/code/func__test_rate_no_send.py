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
def _test_rate_no_send(cl, proto):
    cl.incr('foo', rate=0.5)
    _sock_check(cl._sock, 0, proto)
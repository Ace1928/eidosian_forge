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
def _test_ipv6(cl, proto, addr):
    cl.gauge('foo', 30)
    _sock_check(cl._sock, 1, proto, 'foo:30|g', addr=addr)
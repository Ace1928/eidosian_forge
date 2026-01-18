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
def _test_prefix(cl, proto):
    cl.incr('bar')
    _sock_check(cl._sock, 1, proto, 'foo.bar:1|c')
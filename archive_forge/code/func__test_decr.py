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
def _test_decr(cl, proto):
    cl.decr('foo')
    _sock_check(cl._sock, 1, proto, 'foo:-1|c')
    cl.decr('foo', 10)
    _sock_check(cl._sock, 2, proto, 'foo:-10|c')
    cl.decr('foo', 1.2)
    _sock_check(cl._sock, 3, proto, 'foo:-1.2|c')
    cl.decr('foo', 1, rate=0.5)
    _sock_check(cl._sock, 4, proto, 'foo:-1|c|@0.5')
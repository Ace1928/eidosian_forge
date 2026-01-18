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
def _test_pipeline_manager(cl, proto):
    with cl.pipeline() as pipe:
        pipe.incr('foo')
        pipe.decr('bar')
        pipe.gauge('baz', 15)
    _sock_check(cl._sock, 1, proto, 'foo:1|c\nbar:-1|c\nbaz:15|g')
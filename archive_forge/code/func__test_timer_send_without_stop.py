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
def _test_timer_send_without_stop(cl):
    with cl.timer('foo') as t:
        assert t.ms is None
        with assert_raises(RuntimeError):
            t.send()
    t = cl.timer('bar').start()
    assert t.ms is None
    with assert_raises(RuntimeError):
        t.send()
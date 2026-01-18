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
def _test_pipeline_empty(cl):
    with cl.pipeline() as pipe:
        pipe.incr('foo')
        eq_(1, len(pipe._stats))
    eq_(0, len(pipe._stats))
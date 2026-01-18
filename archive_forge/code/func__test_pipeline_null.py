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
def _test_pipeline_null(cl, proto):
    pipe = cl.pipeline()
    pipe.send()
    _sock_check(cl._sock, 0, proto)
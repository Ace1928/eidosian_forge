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
def _sock_check(sock, count, proto, val=None, addr=None):
    send = send_method[proto](sock)
    eq_(send.call_count, count)
    if not addr:
        addr = ADDR
    if val is not None:
        eq_(send.call_args, make_val[proto](val, addr))
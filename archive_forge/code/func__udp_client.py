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
def _udp_client(prefix=None, addr=None, port=None, ipv6=False):
    if not addr:
        addr = ADDR[0]
    if not port:
        port = ADDR[1]
    sc = StatsClient(host=addr, port=port, prefix=prefix, ipv6=ipv6)
    sc._sock = mock.Mock()
    return sc
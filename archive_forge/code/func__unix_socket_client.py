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
def _unix_socket_client(prefix=None, socket_path=None):
    if not socket_path:
        socket_path = UNIX_SOCKET
    sc = UnixSocketStatsClient(socket_path=socket_path, prefix=prefix)
    sc._sock = mock.Mock()
    return sc
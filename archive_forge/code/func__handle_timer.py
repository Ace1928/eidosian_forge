import copy
import functools
import socket
import struct
import time
from typing import Any, Optional
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import dns.inet
def _handle_timer(self, expiration):
    now = time.time()
    if expiration <= now:
        self._connection.handle_timer(now)
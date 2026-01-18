import copy
import functools
import socket
import struct
import time
from typing import Any, Optional
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import dns.inet
def _common_add_input(self, data, is_end):
    self._buffer.put(data, is_end)
    try:
        return self._expecting > 0 and self._buffer.have(self._expecting)
    except UnexpectedEOF:
        return True
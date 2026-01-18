import copy
import functools
import socket
import struct
import time
from typing import Any, Optional
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import dns.inet
def _expiration_from_timeout(self, timeout):
    if timeout is not None:
        expiration = time.time() + timeout
    else:
        expiration = None
    return expiration
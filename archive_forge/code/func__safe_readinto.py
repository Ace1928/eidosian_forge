import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
def _safe_readinto(self, b):
    """Same as _safe_read, but for reading into a buffer."""
    amt = len(b)
    n = self.fp.readinto(b)
    if n < amt:
        raise IncompleteRead(bytes(b[:n]), amt - n)
    return n
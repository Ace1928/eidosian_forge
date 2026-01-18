import contextlib
import os
import platform
import socket
import sys
import textwrap
import typing  # noqa: F401
import unittest
import warnings
from tornado.testing import bind_unused_port
def _detect_ipv6():
    if not socket.has_ipv6:
        return False
    sock = None
    try:
        sock = socket.socket(socket.AF_INET6)
        sock.bind(('::1', 0))
    except socket.error:
        return False
    finally:
        if sock is not None:
            sock.close()
    return True
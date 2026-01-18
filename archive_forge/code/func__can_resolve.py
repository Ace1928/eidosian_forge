import errno
import logging
import os
import platform
import socket
import ssl
import sys
import warnings
import pytest
from urllib3 import util
from urllib3.exceptions import HTTPWarning
from urllib3.packages import six
from urllib3.util import ssl_
def _can_resolve(host):
    """Returns True if the system can resolve host to an address."""
    try:
        socket.getaddrinfo(host, None, socket.AF_UNSPEC)
        return True
    except socket.gaierror:
        return False
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
def _has_route():
    try:
        sock = socket.create_connection((TARPIT_HOST, 80), 0.0001)
        sock.close()
        return True
    except socket.timeout:
        return True
    except socket.error as e:
        if _is_unreachable_err(e):
            return False
        else:
            raise
import errno
import socket
import time
import logging
import traceback as traceback_
from collections import namedtuple
import http.client
import urllib.request
import pytest
from jaraco.text import trim, unwrap
from cheroot.test import helper, webtest
from cheroot._compat import IS_CI, IS_MACOS, IS_PYPY, IS_WINDOWS
import cheroot.server
class FaultySelect:
    """Mock class to insert errors in the selector.select method."""

    def __init__(self, original_select):
        """Initilize helper class to wrap the selector.select method."""
        self.original_select = original_select
        self.request_served = False
        self.os_error_triggered = False

    def __call__(self, timeout):
        """Intercept the calls to selector.select."""
        if self.request_served:
            self.os_error_triggered = True
            raise OSError('Error while selecting the client socket.')
        return self.original_select(timeout)
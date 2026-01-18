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
class FaultyGetMap:
    """Mock class to insert errors in the selector.get_map method."""

    def __init__(self, original_get_map):
        """Initilize helper class to wrap the selector.get_map method."""
        self.original_get_map = original_get_map
        self.sabotage_conn = False
        self.conn_closed = False

    def __call__(self):
        """Intercept the calls to selector.get_map."""
        sabotage_targets = (conn for _, (_, _, _, conn) in self.original_get_map().items() if isinstance(conn, cheroot.server.HTTPConnection)) if self.sabotage_conn and (not self.conn_closed) else ()
        for conn in sabotage_targets:
            conn.close()
            self.conn_closed = True
        return self.original_get_map()
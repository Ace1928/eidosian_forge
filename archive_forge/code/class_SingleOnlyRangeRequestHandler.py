import io
import socket
import sys
import threading
from http.client import UnknownProtocol, parse_headers
from http.server import SimpleHTTPRequestHandler
import breezy
from .. import (config, controldir, debug, errors, osutils, tests, trace,
from ..bzr import remote as _mod_remote
from ..transport import remote
from ..transport.http import urllib
from ..transport.http.urllib import (AbstractAuthHandler, BasicAuthHandler,
from . import features, http_server, http_utils, test_server
from .scenarios import load_tests_apply_scenarios, multiply_scenarios
class SingleOnlyRangeRequestHandler(http_server.TestingHTTPRequestHandler):
    """Only reply to simple range requests, errors out on multiple"""

    def get_multiple_ranges(self, file, file_size, ranges):
        """Refuses the multiple ranges request"""
        if len(ranges) > 1:
            file.close()
            self.send_error(416, 'Requested range not satisfiable')
            return
        start, end = ranges[0]
        return self.get_single_range(file, file_size, start, end)
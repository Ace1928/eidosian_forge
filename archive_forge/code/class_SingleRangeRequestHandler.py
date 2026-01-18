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
class SingleRangeRequestHandler(http_server.TestingHTTPRequestHandler):
    """Always reply to range request as if they were single.

    Don't be explicit about it, just to annoy the clients.
    """

    def get_multiple_ranges(self, file, file_size, ranges):
        """Answer as if it was a single range request and ignores the rest"""
        start, end = ranges[0]
        return self.get_single_range(file, file_size, start, end)
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
class TestRangeHeader(tests.TestCase):
    """Test range_header method"""

    def check_header(self, value, ranges=[], tail=0):
        offsets = [(start, end - start + 1) for start, end in ranges]
        coalesce = transport.Transport._coalesce_offsets
        coalesced = list(coalesce(offsets, limit=0, fudge_factor=0))
        range_header = HttpTransport._range_header
        self.assertEqual(value, range_header(coalesced, tail))

    def test_range_header_single(self):
        self.check_header('0-9', ranges=[(0, 9)])
        self.check_header('100-109', ranges=[(100, 109)])

    def test_range_header_tail(self):
        self.check_header('-10', tail=10)
        self.check_header('-50', tail=50)

    def test_range_header_multi(self):
        self.check_header('0-9,100-200,300-5000', ranges=[(0, 9), (100, 200), (300, 5000)])

    def test_range_header_mixed(self):
        self.check_header('0-9,300-5000,-50', ranges=[(0, 9), (300, 5000)], tail=50)
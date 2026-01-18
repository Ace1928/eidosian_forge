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
class TestHTTPRangeParsing(tests.TestCase):

    def setUp(self):
        super().setUp()

        class RequestHandler(http_server.TestingHTTPRequestHandler):

            def setup(self):
                pass

            def handle(self):
                pass

            def finish(self):
                pass
        self.req_handler = RequestHandler(None, None, None)

    def assertRanges(self, ranges, header, file_size):
        self.assertEqual(ranges, self.req_handler._parse_ranges(header, file_size))

    def test_simple_range(self):
        self.assertRanges([(0, 2)], 'bytes=0-2', 12)

    def test_tail(self):
        self.assertRanges([(8, 11)], 'bytes=-4', 12)

    def test_tail_bigger_than_file(self):
        self.assertRanges([(0, 11)], 'bytes=-99', 12)

    def test_range_without_end(self):
        self.assertRanges([(4, 11)], 'bytes=4-', 12)

    def test_invalid_ranges(self):
        self.assertRanges(None, 'bytes=12-22', 12)
        self.assertRanges(None, 'bytes=1-3,12-22', 12)
        self.assertRanges(None, 'bytes=-', 12)
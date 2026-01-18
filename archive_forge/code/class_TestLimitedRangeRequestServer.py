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
class TestLimitedRangeRequestServer(http_utils.TestCaseWithWebserver):
    """Tests readv requests against a server erroring out on too much ranges."""
    scenarios = multiply_scenarios(vary_by_http_client_implementation(), vary_by_http_protocol_version())
    range_limit = 3

    def create_transport_readonly_server(self):
        return LimitedRangeHTTPServer(range_limit=self.range_limit, protocol_version=self._protocol_version)

    def setUp(self):
        super().setUp()
        filler = b''.join([b'abcdefghij' for x in range(102)])
        content = b''.join([b'%04d' % v + filler for v in range(16)])
        self.build_tree_contents([('a', content)])

    def test_few_ranges(self):
        t = self.get_readonly_transport()
        l = list(t.readv('a', ((0, 4), (1024, 4))))
        self.assertEqual(l[0], (0, b'0000'))
        self.assertEqual(l[1], (1024, b'0001'))
        self.assertEqual(1, self.get_readonly_server().GET_request_nb)

    def test_more_ranges(self):
        t = self.get_readonly_transport()
        l = list(t.readv('a', ((0, 4), (1024, 4), (4096, 4), (8192, 4))))
        self.assertEqual(l[0], (0, b'0000'))
        self.assertEqual(l[1], (1024, b'0001'))
        self.assertEqual(l[2], (4096, b'0004'))
        self.assertEqual(l[3], (8192, b'0008'))
        self.assertEqual(2, self.get_readonly_server().GET_request_nb)
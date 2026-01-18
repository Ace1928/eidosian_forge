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
class TestTruncatedMultipleRangeServer(TestSpecificRequestHandler):
    _req_handler_class = TruncatedMultipleRangeRequestHandler

    def setUp(self):
        super().setUp()
        self.build_tree_contents([('a', b'0123456789')])

    def test_readv_with_short_reads(self):
        server = self.get_readonly_server()
        t = self.get_readonly_transport()
        t._bytes_to_read_before_seek = 0
        ireadv = iter(t.readv('a', ((0, 1), (2, 1), (4, 2), (9, 1))))
        self.assertEqual((0, b'0'), next(ireadv))
        self.assertEqual((2, b'2'), next(ireadv))
        self.assertEqual(1, server.GET_request_nb)
        self.assertEqual((4, b'45'), next(ireadv))
        self.assertEqual((9, b'9'), next(ireadv))
        self.assertEqual(3, server.GET_request_nb)
        self.assertEqual('single', t._range_hint)
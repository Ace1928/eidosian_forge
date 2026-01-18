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
class TestHTTPConnections(http_utils.TestCaseWithWebserver):
    """Test the http connections."""
    scenarios = multiply_scenarios(vary_by_http_client_implementation(), vary_by_http_protocol_version())

    def setUp(self):
        super().setUp()
        self.build_tree(['foo/', 'foo/bar'], line_endings='binary', transport=self.get_transport())

    def test_http_has(self):
        server = self.get_readonly_server()
        t = self.get_readonly_transport()
        self.assertEqual(t.has('foo/bar'), True)
        self.assertEqual(len(server.logs), 1)
        self.assertContainsRe(server.logs[0], '"HEAD /foo/bar HTTP/1.." (200|302) - "-" "Breezy/')

    def test_http_has_not_found(self):
        server = self.get_readonly_server()
        t = self.get_readonly_transport()
        self.assertEqual(t.has('not-found'), False)
        self.assertContainsRe(server.logs[1], '"HEAD /not-found HTTP/1.." 404 - "-" "Breezy/')

    def test_http_get(self):
        server = self.get_readonly_server()
        t = self.get_readonly_transport()
        fp = t.get('foo/bar')
        self.assertEqualDiff(fp.read(), b'contents of foo/bar\n')
        self.assertEqual(len(server.logs), 1)
        self.assertTrue(server.logs[0].find('"GET /foo/bar HTTP/1.1" 200 - "-" "Breezy/%s' % breezy.__version__) > -1)

    def test_has_on_bogus_host(self):
        default_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(2)
            s = socket.socket()
            s.bind(('localhost', 0))
            t = self._transport('http://%s:%s/' % s.getsockname())
            self.assertRaises(errors.ConnectionError, t.has, 'foo/bar')
        finally:
            socket.setdefaulttimeout(default_timeout)
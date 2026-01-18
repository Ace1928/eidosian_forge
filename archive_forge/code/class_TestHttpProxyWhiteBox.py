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
class TestHttpProxyWhiteBox(tests.TestCase):
    """Whitebox test proxy http authorization.

    Only the urllib implementation is tested here.
    """

    def _proxied_request(self):
        handler = ProxyHandler()
        request = Request('GET', 'http://baz/buzzle')
        handler.set_proxy(request, 'http')
        return request

    def assertEvaluateProxyBypass(self, expected, host, no_proxy):
        handler = ProxyHandler()
        self.assertEqual(expected, handler.evaluate_proxy_bypass(host, no_proxy))

    def test_empty_user(self):
        self.overrideEnv('http_proxy', 'http://bar.com')
        request = self._proxied_request()
        self.assertFalse('Proxy-authorization' in request.headers)

    def test_user_with_at(self):
        self.overrideEnv('http_proxy', 'http://username@domain:password@proxy_host:1234')
        request = self._proxied_request()
        self.assertFalse('Proxy-authorization' in request.headers)

    def test_invalid_proxy(self):
        """A proxy env variable without scheme"""
        self.overrideEnv('http_proxy', 'host:1234')
        self.assertRaises(urlutils.InvalidURL, self._proxied_request)

    def test_evaluate_proxy_bypass_true(self):
        """The host is not proxied"""
        self.assertEvaluateProxyBypass(True, 'example.com', 'example.com')
        self.assertEvaluateProxyBypass(True, 'bzr.example.com', '*example.com')

    def test_evaluate_proxy_bypass_false(self):
        """The host is proxied"""
        self.assertEvaluateProxyBypass(False, 'bzr.example.com', None)

    def test_evaluate_proxy_bypass_unknown(self):
        """The host is not explicitly proxied"""
        self.assertEvaluateProxyBypass(None, 'example.com', 'not.example.com')
        self.assertEvaluateProxyBypass(None, 'bzr.example.com', 'example.com')

    def test_evaluate_proxy_bypass_empty_entries(self):
        """Ignore empty entries"""
        self.assertEvaluateProxyBypass(None, 'example.com', '')
        self.assertEvaluateProxyBypass(None, 'example.com', ',')
        self.assertEvaluateProxyBypass(None, 'example.com', 'foo,,bar')
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
class TestProxyHttpServer(http_utils.TestCaseWithTwoWebservers):
    """Tests proxy server.

    Be aware that we do not setup a real proxy here. Instead, we
    check that the *connection* goes through the proxy by serving
    different content (the faked proxy server append '-proxied'
    to the file names).
    """
    scenarios = multiply_scenarios(vary_by_http_client_implementation(), vary_by_http_protocol_version())

    def setUp(self):
        super().setUp()
        self.transport_secondary_server = http_utils.ProxyServer
        self.build_tree_contents([('foo', b'contents of foo\n'), ('foo-proxied', b'proxied contents of foo\n')])
        server = self.get_readonly_server()
        self.server_host_port = '%s:%d' % (server.host, server.port)
        self.no_proxy_host = self.server_host_port
        self.proxy_url = self.get_secondary_url()

    def assertProxied(self):
        t = self.get_readonly_transport()
        self.assertEqual(b'proxied contents of foo\n', t.get('foo').read())

    def assertNotProxied(self):
        t = self.get_readonly_transport()
        self.assertEqual(b'contents of foo\n', t.get('foo').read())

    def test_http_proxy(self):
        self.overrideEnv('http_proxy', self.proxy_url)
        self.assertProxied()

    def test_HTTP_PROXY(self):
        self.overrideEnv('HTTP_PROXY', self.proxy_url)
        self.assertProxied()

    def test_all_proxy(self):
        self.overrideEnv('all_proxy', self.proxy_url)
        self.assertProxied()

    def test_ALL_PROXY(self):
        self.overrideEnv('ALL_PROXY', self.proxy_url)
        self.assertProxied()

    def test_http_proxy_with_no_proxy(self):
        self.overrideEnv('no_proxy', self.no_proxy_host)
        self.overrideEnv('http_proxy', self.proxy_url)
        self.assertNotProxied()

    def test_HTTP_PROXY_with_NO_PROXY(self):
        self.overrideEnv('NO_PROXY', self.no_proxy_host)
        self.overrideEnv('HTTP_PROXY', self.proxy_url)
        self.assertNotProxied()

    def test_all_proxy_with_no_proxy(self):
        self.overrideEnv('no_proxy', self.no_proxy_host)
        self.overrideEnv('all_proxy', self.proxy_url)
        self.assertNotProxied()

    def test_ALL_PROXY_with_NO_PROXY(self):
        self.overrideEnv('NO_PROXY', self.no_proxy_host)
        self.overrideEnv('ALL_PROXY', self.proxy_url)
        self.assertNotProxied()

    def test_http_proxy_without_scheme(self):
        self.overrideEnv('http_proxy', self.server_host_port)
        self.assertRaises(urlutils.InvalidURL, self.assertProxied)
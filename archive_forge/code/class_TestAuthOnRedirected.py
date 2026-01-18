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
class TestAuthOnRedirected(http_utils.TestCaseWithRedirectedWebserver):
    """Test authentication on the redirected http server."""
    scenarios = vary_by_http_protocol_version()
    _auth_header = 'Authorization'
    _password_prompt_prefix = ''
    _username_prompt_prefix = ''
    _auth_server = http_utils.HTTPBasicAuthServer
    _transport = HttpTransport

    def setUp(self):
        super().setUp()
        self.build_tree_contents([('a', b'a'), ('1/',), ('1/a', b'redirected once')])
        new_prefix = 'http://{}:{}'.format(self.new_server.host, self.new_server.port)
        self.old_server.redirections = [('(.*)', '%s/1\\1' % new_prefix, 301)]
        self.old_transport = self.get_old_transport()
        self.new_server.add_user('joe', 'foo')
        cleanup_http_redirection_connections(self)

    def create_transport_readonly_server(self):
        server = self._auth_server(protocol_version=self._protocol_version)
        server._url_protocol = self._url_protocol
        return server

    def get_a(self, t):
        return t.get('a')

    def test_auth_on_redirected_via_do_catching_redirections(self):
        self.redirections = 0

        def redirected(t, exception, redirection_notice):
            self.redirections += 1
            redirected_t = t._redirected_to(exception.source, exception.target)
            self.addCleanup(redirected_t.disconnect)
            return redirected_t
        ui.ui_factory = tests.TestUIFactory(stdin='joe\nfoo\n')
        self.assertEqual(b'redirected once', transport.do_catching_redirections(self.get_a, self.old_transport, redirected).read())
        self.assertEqual(1, self.redirections)
        self.assertEqual('', ui.ui_factory.stdin.readline())
        self.assertEqual('', ui.ui_factory.stdout.getvalue())

    def test_auth_on_redirected_via_following_redirections(self):
        self.new_server.add_user('joe', 'foo')
        ui.ui_factory = tests.TestUIFactory(stdin='joe\nfoo\n')
        t = self.old_transport
        new_prefix = 'http://{}:{}'.format(self.new_server.host, self.new_server.port)
        self.old_server.redirections = [('(.*)', '%s/1\\1' % new_prefix, 301)]
        self.assertEqual(b'redirected once', t.request('GET', t.abspath('a'), retries=3).read())
        self.assertEqual('', ui.ui_factory.stdin.readline())
        self.assertEqual('', ui.ui_factory.stdout.getvalue())
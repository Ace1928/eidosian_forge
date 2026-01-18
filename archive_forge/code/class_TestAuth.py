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
class TestAuth(http_utils.TestCaseWithWebserver):
    """Test authentication scheme"""
    scenarios = multiply_scenarios(vary_by_http_client_implementation(), vary_by_http_protocol_version(), vary_by_http_auth_scheme())

    def setUp(self):
        super().setUp()
        self.server = self.get_readonly_server()
        self.build_tree_contents([('a', b'contents of a\n'), ('b', b'contents of b\n')])

    def create_transport_readonly_server(self):
        server = self._auth_server(protocol_version=self._protocol_version)
        server._url_protocol = self._url_protocol
        return server

    def get_user_url(self, user, password):
        """Build an url embedding user and password"""
        url = '%s://' % self.server._url_protocol
        if user is not None:
            url += user
            if password is not None:
                url += ':' + password
            url += '@'
        url += '{}:{}/'.format(self.server.host, self.server.port)
        return url

    def get_user_transport(self, user, password):
        t = transport.get_transport_from_url(self.get_user_url(user, password))
        return t

    def test_no_user(self):
        self.server.add_user('joe', 'foo')
        t = self.get_user_transport(None, None)
        self.assertRaises(errors.InvalidHttpResponse, t.get, 'a')
        self.assertEqual(1, self.server.auth_required_errors)

    def test_empty_pass(self):
        self.server.add_user('joe', '')
        t = self.get_user_transport('joe', '')
        self.assertEqual(b'contents of a\n', t.get('a').read())
        self.assertEqual(1, self.server.auth_required_errors)

    def test_user_pass(self):
        self.server.add_user('joe', 'foo')
        t = self.get_user_transport('joe', 'foo')
        self.assertEqual(b'contents of a\n', t.get('a').read())
        self.assertEqual(1, self.server.auth_required_errors)

    def test_unknown_user(self):
        self.server.add_user('joe', 'foo')
        t = self.get_user_transport('bill', 'foo')
        self.assertRaises(errors.InvalidHttpResponse, t.get, 'a')
        self.assertEqual(2, self.server.auth_required_errors)

    def test_wrong_pass(self):
        self.server.add_user('joe', 'foo')
        t = self.get_user_transport('joe', 'bar')
        self.assertRaises(errors.InvalidHttpResponse, t.get, 'a')
        self.assertEqual(2, self.server.auth_required_errors)

    def test_prompt_for_username(self):
        self.server.add_user('joe', 'foo')
        t = self.get_user_transport(None, None)
        ui.ui_factory = tests.TestUIFactory(stdin='joe\nfoo\n')
        stdout, stderr = (ui.ui_factory.stdout, ui.ui_factory.stderr)
        self.assertEqual(b'contents of a\n', t.get('a').read())
        self.assertEqual('', ui.ui_factory.stdin.readline())
        stderr.seek(0)
        expected_prompt = self._expected_username_prompt(t._unqualified_scheme)
        self.assertEqual(expected_prompt, stderr.read(len(expected_prompt)))
        self.assertEqual('', stdout.getvalue())
        self._check_password_prompt(t._unqualified_scheme, 'joe', stderr.readline())

    def test_prompt_for_password(self):
        self.server.add_user('joe', 'foo')
        t = self.get_user_transport('joe', None)
        ui.ui_factory = tests.TestUIFactory(stdin='foo\n')
        stdout, stderr = (ui.ui_factory.stdout, ui.ui_factory.stderr)
        self.assertEqual(b'contents of a\n', t.get('a').read())
        self.assertEqual('', ui.ui_factory.stdin.readline())
        self._check_password_prompt(t._unqualified_scheme, 'joe', stderr.getvalue())
        self.assertEqual('', stdout.getvalue())
        self.assertEqual(b'contents of b\n', t.get('b').read())
        t2 = t.clone()
        self.assertEqual(b'contents of b\n', t2.get('b').read())
        self.assertEqual(1, self.server.auth_required_errors)

    def _check_password_prompt(self, scheme, user, actual_prompt):
        expected_prompt = self._password_prompt_prefix + "%s %s@%s:%d, Realm: '%s' password: " % (scheme.upper(), user, self.server.host, self.server.port, self.server.auth_realm)
        self.assertEqual(expected_prompt, actual_prompt)

    def _expected_username_prompt(self, scheme):
        return self._username_prompt_prefix + "%s %s:%d, Realm: '%s' username: " % (scheme.upper(), self.server.host, self.server.port, self.server.auth_realm)

    def test_no_prompt_for_password_when_using_auth_config(self):
        user = ' joe'
        password = 'foo'
        stdin_content = 'bar\n'
        self.server.add_user(user, password)
        t = self.get_user_transport(user, None)
        ui.ui_factory = tests.TestUIFactory(stdin=stdin_content)
        _setup_authentication_config(scheme='http', port=self.server.port, user=user, password=password)
        with t.get('a') as f:
            self.assertEqual(b'contents of a\n', f.read())
        self.assertEqual(stdin_content, ui.ui_factory.stdin.readline())
        self.assertEqual(1, self.server.auth_required_errors)

    def test_changing_nonce(self):
        if self._auth_server not in (http_utils.HTTPDigestAuthServer, http_utils.ProxyDigestAuthServer):
            raise tests.TestNotApplicable('HTTP/proxy auth digest only test')
        self.server.add_user('joe', 'foo')
        t = self.get_user_transport('joe', 'foo')
        with t.get('a') as f:
            self.assertEqual(b'contents of a\n', f.read())
        with t.get('b') as f:
            self.assertEqual(b'contents of b\n', f.read())
        self.assertEqual(1, self.server.auth_required_errors)
        self.server.auth_nonce = self.server.auth_nonce + '. No, now!'
        self.assertEqual(b'contents of a\n', t.get('a').read())
        self.assertEqual(2, self.server.auth_required_errors)

    def test_user_from_auth_conf(self):
        user = 'joe'
        password = 'foo'
        self.server.add_user(user, password)
        _setup_authentication_config(scheme='http', port=self.server.port, user=user, password=password)
        t = self.get_user_transport(None, None)
        with t.get('a') as f:
            self.assertEqual(b'contents of a\n', f.read())
        self.assertEqual(1, self.server.auth_required_errors)

    def test_no_credential_leaks_in_log(self):
        self.overrideAttr(debug, 'debug_flags', {'http'})
        user = 'joe'
        password = 'very-sensitive-password'
        self.server.add_user(user, password)
        t = self.get_user_transport(user, password)
        self.mutters = []

        def mutter(*args):
            lines = args[0] % args[1:]
            self.mutters.extend(lines.splitlines())
        self.overrideAttr(trace, 'mutter', mutter)
        self.assertEqual(True, t.has('a'))
        self.assertEqual(1, self.server.auth_required_errors)
        sent_auth_headers = [line for line in self.mutters if line.startswith('> {}'.format(self._auth_header))]
        self.assertLength(1, sent_auth_headers)
        self.assertStartsWith(sent_auth_headers[0], '> {}: <masked>'.format(self._auth_header))
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
class TestAuthHeader(tests.TestCase):

    def parse_header(self, header, auth_handler_class=None):
        if auth_handler_class is None:
            auth_handler_class = AbstractAuthHandler
        self.auth_handler = auth_handler_class()
        return self.auth_handler._parse_auth_header(header)

    def test_empty_header(self):
        scheme, remainder = self.parse_header('')
        self.assertEqual('', scheme)
        self.assertIs(None, remainder)

    def test_negotiate_header(self):
        scheme, remainder = self.parse_header('Negotiate')
        self.assertEqual('negotiate', scheme)
        self.assertIs(None, remainder)

    def test_basic_header(self):
        scheme, remainder = self.parse_header('Basic realm="Thou should not pass"')
        self.assertEqual('basic', scheme)
        self.assertEqual('realm="Thou should not pass"', remainder)

    def test_build_basic_header_with_long_creds(self):
        handler = BasicAuthHandler()
        user = 'user' * 10
        password = 'password' * 5
        header = handler.build_auth_header(dict(user=user, password=password), None)
        self.assertFalse('\n' in header)

    def test_basic_extract_realm(self):
        scheme, remainder = self.parse_header('Basic realm="Thou should not pass"', BasicAuthHandler)
        match, realm = self.auth_handler.extract_realm(remainder)
        self.assertTrue(match is not None)
        self.assertEqual('Thou should not pass', realm)

    def test_digest_header(self):
        scheme, remainder = self.parse_header('Digest realm="Thou should not pass"')
        self.assertEqual('digest', scheme)
        self.assertEqual('realm="Thou should not pass"', remainder)
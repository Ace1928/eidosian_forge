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
class TestBadProtocolServer(TestSpecificRequestHandler):
    """Tests bad protocol from server."""
    _req_handler_class = BadProtocolRequestHandler

    def test_http_has(self):
        t = self.get_readonly_transport()
        self.assertRaises(errors.InvalidHttpResponse, t.has, 'foo/bar')

    def test_http_get(self):
        t = self.get_readonly_transport()
        self.assertRaises(errors.InvalidHttpResponse, t.get, 'foo/bar')
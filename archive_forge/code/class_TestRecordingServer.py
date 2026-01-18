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
class TestRecordingServer(tests.TestCase):

    def test_create(self):
        server = RecordingServer(expect_body_tail=None)
        self.assertEqual(b'', server.received_bytes)
        self.assertEqual(None, server.host)
        self.assertEqual(None, server.port)

    def test_setUp_and_stop(self):
        server = RecordingServer(expect_body_tail=None)
        server.start_server()
        try:
            self.assertNotEqual(None, server.host)
            self.assertNotEqual(None, server.port)
        finally:
            server.stop_server()
        self.assertEqual(None, server.host)
        self.assertEqual(None, server.port)

    def test_send_receive_bytes(self):
        server = RecordingServer(expect_body_tail=b'c', scheme='http')
        self.start_server(server)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server.host, server.port))
        sock.sendall(b'abc')
        self.assertEqual(b'HTTP/1.1 200 OK\r\n', osutils.recv_all(sock, 4096))
        self.assertEqual(b'abc', server.received_bytes)
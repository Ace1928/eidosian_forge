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
class SmartHTTPTunnellingTest(tests.TestCaseWithTransport):
    scenarios = multiply_scenarios(vary_by_http_client_implementation(), vary_by_http_protocol_version())

    def setUp(self):
        super().setUp()
        self.overrideEnv('BRZ_NO_SMART_VFS', None)
        self.transport_readonly_server = http_utils.HTTPServerWithSmarts
        self.http_server = self.get_readonly_server()

    def create_transport_readonly_server(self):
        server = http_utils.HTTPServerWithSmarts(protocol_version=self._protocol_version)
        server._url_protocol = self._url_protocol
        return server

    def test_open_controldir(self):
        branch = self.make_branch('relpath')
        url = self.http_server.get_url() + 'relpath'
        bd = controldir.ControlDir.open(url)
        self.addCleanup(bd.transport.disconnect)
        self.assertIsInstance(bd, _mod_remote.RemoteBzrDir)

    def test_bulk_data(self):
        self.build_tree(['data-file'])
        http_transport = transport.get_transport_from_url(self.http_server.get_url())
        medium = http_transport.get_smart_medium()
        remote_transport = remote.RemoteTransport('bzr://fake_host/', medium=medium)
        self.assertEqual([(0, b'c')], list(remote_transport.readv('data-file', [(0, 1)])))

    def test_http_send_smart_request(self):
        post_body = b'hello\n'
        expected_reply_body = b'ok\x012\n'
        http_transport = transport.get_transport_from_url(self.http_server.get_url())
        medium = http_transport.get_smart_medium()
        response = medium.send_http_smart_request(post_body)
        reply_body = response.read()
        self.assertEqual(expected_reply_body, reply_body)

    def test_smart_http_server_post_request_handler(self):
        httpd = self.http_server.server
        socket = SampleSocket(b'POST /.bzr/smart %s \r\n' % self._protocol_version.encode('ascii') + b'Content-Length: 6\r\n\r\nhello\n')
        request_handler = http_utils.SmartRequestHandler(socket, ('localhost', 80), httpd)
        response = socket.writefile.getvalue()
        self.assertStartsWith(response, b'%s 200 ' % self._protocol_version.encode('ascii'))
        expected_end_of_response = b'\r\n\r\nok\x012\n'
        self.assertEndsWith(response, expected_end_of_response)
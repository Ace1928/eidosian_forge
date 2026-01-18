import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
class Test_SmartClientRequest(tests.TestCase):

    def make_client_with_failing_medium(self, fail_at_write=True, response=b''):
        response_io = BytesIO(response)
        output = BytesIO()
        vendor = FirstRejectedBytesIOSSHVendor(response_io, output, fail_at_write=fail_at_write)
        ssh_params = medium.SSHParams('a host', 'a port', 'a user', 'a pass')
        client_medium = medium.SmartSSHClientMedium('base', ssh_params, vendor)
        smart_client = client._SmartClient(client_medium, headers={})
        return (output, vendor, smart_client)

    def make_response(self, args, body=None, body_stream=None):
        response_io = BytesIO()
        response = _mod_request.SuccessfulSmartServerResponse(args, body=body, body_stream=body_stream)
        responder = protocol.ProtocolThreeResponder(response_io.write)
        responder.send_response(response)
        return response_io.getvalue()

    def test__call_doesnt_retry_append(self):
        response = self.make_response(('appended', b'8'))
        output, vendor, smart_client = self.make_client_with_failing_medium(fail_at_write=False, response=response)
        smart_request = client._SmartClientRequest(smart_client, b'append', (b'foo', b''), body=b'content\n')
        self.assertRaises(errors.ConnectionReset, smart_request._call, 3)

    def test__call_retries_get_bytes(self):
        response = self.make_response((b'ok',), b'content\n')
        output, vendor, smart_client = self.make_client_with_failing_medium(fail_at_write=False, response=response)
        smart_request = client._SmartClientRequest(smart_client, b'get', (b'foo',))
        response, response_handler = smart_request._call(3)
        self.assertEqual((b'ok',), response)
        self.assertEqual(b'content\n', response_handler.read_body_bytes())

    def test__call_noretry_get_bytes(self):
        debug.debug_flags.add('noretry')
        response = self.make_response((b'ok',), b'content\n')
        output, vendor, smart_client = self.make_client_with_failing_medium(fail_at_write=False, response=response)
        smart_request = client._SmartClientRequest(smart_client, b'get', (b'foo',))
        self.assertRaises(errors.ConnectionReset, smart_request._call, 3)

    def test__send_no_retry_pipes(self):
        client_read, server_write = create_file_pipes()
        server_read, client_write = create_file_pipes()
        client_medium = medium.SmartSimplePipesClientMedium(client_read, client_write, base='/')
        smart_client = client._SmartClient(client_medium)
        smart_request = client._SmartClientRequest(smart_client, b'hello', ())
        server_read.close()
        encoder, response_handler = smart_request._construct_protocol(3)
        self.assertRaises(errors.ConnectionReset, smart_request._send_no_retry, encoder)

    def test__send_read_response_sockets(self):
        listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.bind(('127.0.0.1', 0))
        listen_sock.listen(1)
        host, port = listen_sock.getsockname()
        client_medium = medium.SmartTCPClientMedium(host, port, '/')
        client_medium._ensure_connection()
        smart_client = client._SmartClient(client_medium)
        smart_request = client._SmartClientRequest(smart_client, b'hello', ())
        server_sock, _ = listen_sock.accept()
        server_sock.close()
        handler = smart_request._send(3)
        self.assertRaises(errors.ConnectionReset, handler.read_response_tuple, expect_body=False)

    def test__send_retries_on_write(self):
        output, vendor, smart_client = self.make_client_with_failing_medium()
        smart_request = client._SmartClientRequest(smart_client, b'hello', ())
        handler = smart_request._send(3)
        self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02des\x00\x00\x00\tl5:helloee', output.getvalue())
        self.assertEqual([('connect_ssh', 'a user', 'a pass', 'a host', 'a port', ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes']), ('close',), ('connect_ssh', 'a user', 'a pass', 'a host', 'a port', ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes'])], vendor.calls)

    def test__send_doesnt_retry_read_failure(self):
        output, vendor, smart_client = self.make_client_with_failing_medium(fail_at_write=False)
        smart_request = client._SmartClientRequest(smart_client, b'hello', ())
        handler = smart_request._send(3)
        self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02des\x00\x00\x00\tl5:helloee', output.getvalue())
        self.assertEqual([('connect_ssh', 'a user', 'a pass', 'a host', 'a port', ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes'])], vendor.calls)
        self.assertRaises(errors.ConnectionReset, handler.read_response_tuple)

    def test__send_request_retries_body_stream_if_not_started(self):
        output, vendor, smart_client = self.make_client_with_failing_medium()
        smart_request = client._SmartClientRequest(smart_client, b'hello', (), body_stream=[b'a', b'b'])
        response_handler = smart_request._send(3)
        self.assertEqual([('connect_ssh', 'a user', 'a pass', 'a host', 'a port', ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes']), ('close',), ('connect_ssh', 'a user', 'a pass', 'a host', 'a port', ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes'])], vendor.calls)
        self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02des\x00\x00\x00\tl5:helloeb\x00\x00\x00\x01ab\x00\x00\x00\x01be', output.getvalue())

    def test__send_request_stops_if_body_started(self):
        from io import BytesIO
        response = BytesIO()

        class FailAfterFirstWrite(BytesIO):
            """Allow one 'write' call to pass, fail the rest"""

            def __init__(self):
                BytesIO.__init__(self)
                self._first = True

            def write(self, s):
                if self._first:
                    self._first = False
                    return BytesIO.write(self, s)
                raise OSError(errno.EINVAL, 'invalid file handle')
        output = FailAfterFirstWrite()
        vendor = FirstRejectedBytesIOSSHVendor(response, output, fail_at_write=False)
        ssh_params = medium.SSHParams('a host', 'a port', 'a user', 'a pass')
        client_medium = medium.SmartSSHClientMedium('base', ssh_params, vendor)
        smart_client = client._SmartClient(client_medium, headers={})
        smart_request = client._SmartClientRequest(smart_client, b'hello', (), body_stream=[b'a', b'b'])
        self.assertRaises(errors.ConnectionReset, smart_request._send, 3)
        self.assertEqual([('connect_ssh', 'a user', 'a pass', 'a host', 'a port', ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes']), ('close',)], vendor.calls)
        self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02des\x00\x00\x00\tl5:helloe', output.getvalue())

    def test__send_disabled_retry(self):
        debug.debug_flags.add('noretry')
        output, vendor, smart_client = self.make_client_with_failing_medium()
        smart_request = client._SmartClientRequest(smart_client, b'hello', ())
        self.assertRaises(errors.ConnectionReset, smart_request._send, 3)
        self.assertEqual([('connect_ssh', 'a user', 'a pass', 'a host', 'a port', ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes']), ('close',)], vendor.calls)
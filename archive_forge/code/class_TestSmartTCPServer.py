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
class TestSmartTCPServer(tests.TestCase):

    def make_server(self):
        """Create a SmartTCPServer that we can exercise.

        Note: we don't use SmartTCPServer_for_testing because the testing
        version overrides lots of functionality like 'serve', and we want to
        test the raw service.

        This will start the server in another thread, and wait for it to
        indicate it has finished starting up.

        :return: (server, server_thread)
        """
        t = _mod_transport.get_transport_from_url('memory:///')
        server = _mod_server.SmartTCPServer(t, client_timeout=4.0)
        server._ACCEPT_TIMEOUT = 0.1
        server.start_server('127.0.0.1', 0)
        server_thread = threading.Thread(target=server.serve, args=(self.id(),))
        server_thread.start()
        self.addCleanup(server._stop_gracefully)
        server._started.wait()
        return (server, server_thread)

    def ensure_client_disconnected(self, client_sock):
        """Ensure that a socket is closed, discarding all errors."""
        try:
            client_sock.close()
        except Exception:
            pass

    def connect_to_server(self, server):
        """Create a client socket that can talk to the server."""
        client_sock = socket.socket()
        server_info = server._server_socket.getsockname()
        client_sock.connect(server_info)
        self.addCleanup(self.ensure_client_disconnected, client_sock)
        return client_sock

    def connect_to_server_and_hangup(self, server):
        """Connect to the server, and then hang up.
        That way it doesn't sit waiting for 'accept()' to timeout.
        """
        if server._stopped.is_set():
            return
        try:
            client_sock = self.connect_to_server(server)
            client_sock.close()
        except OSError as e:
            pass

    def say_hello(self, client_sock):
        """Send the 'hello' smart RPC, and expect the response."""
        client_sock.send(b'hello\n')
        self.assertEqual(b'ok\x012\n', client_sock.recv(5))

    def shutdown_server_cleanly(self, server, server_thread):
        server._stop_gracefully()
        self.connect_to_server_and_hangup(server)
        server._stopped.wait()
        server._fully_stopped.wait()
        server_thread.join()

    def test_get_error_unexpected(self):
        """Error reported by server with no specific representation"""
        self.overrideEnv('BRZ_NO_SMART_VFS', None)

        class FlakyTransport:
            base = 'a_url'

            def external_url(self):
                return self.base

            def get(self, path):
                raise Exception('some random exception from inside server')

        class FlakyServer(test_server.SmartTCPServer_for_testing):

            def get_backing_transport(self, backing_transport_server):
                return FlakyTransport()
        smart_server = FlakyServer()
        smart_server.start_server()
        self.addCleanup(smart_server.stop_server)
        t = remote.RemoteTCPTransport(smart_server.get_url())
        self.addCleanup(t.disconnect)
        err = self.assertRaises(UnknownErrorFromSmartServer, t.get, 'something')
        self.assertContainsRe(str(err), 'some random exception')

    def test_propagates_timeout(self):
        server = _mod_server.SmartTCPServer(None, client_timeout=1.23)
        server_sock, client_sock = portable_socket_pair()
        handler = server._make_handler(server_sock)
        self.assertEqual(1.23, handler._client_timeout)

    def test_serve_conn_tracks_connections(self):
        server = _mod_server.SmartTCPServer(None, client_timeout=4.0)
        server_sock, client_sock = portable_socket_pair()
        server.serve_conn(server_sock, '-{}'.format(self.id()))
        self.assertEqual(1, len(server._active_connections))
        server._poll_active_connections()
        self.assertEqual(1, len(server._active_connections))
        client_sock.close()
        server._poll_active_connections(0.1)
        self.assertEqual(0, len(server._active_connections))

    def test_serve_closes_out_finished_connections(self):
        server, server_thread = self.make_server()
        client_sock = self.connect_to_server(server)
        self.say_hello(client_sock)
        self.assertEqual(1, len(server._active_connections))
        _, server_side_thread = server._active_connections[0]
        client_sock.close()
        server_side_thread.join()
        self.shutdown_server_cleanly(server, server_thread)
        self.assertEqual(0, len(server._active_connections))

    def test_serve_reaps_finished_connections(self):
        server, server_thread = self.make_server()
        client_sock1 = self.connect_to_server(server)
        self.say_hello(client_sock1)
        server_handler1, server_side_thread1 = server._active_connections[0]
        client_sock1.close()
        server_side_thread1.join()
        client_sock2 = self.connect_to_server(server)
        self.say_hello(client_sock2)
        server_handler2, server_side_thread2 = server._active_connections[-1]
        client_sock3 = self.connect_to_server(server)
        self.say_hello(client_sock3)
        conns = list(server._active_connections)
        self.assertEqual(2, len(conns))
        self.assertNotEqual((server_handler1, server_side_thread1), conns[0])
        self.assertEqual((server_handler2, server_side_thread2), conns[0])
        client_sock2.close()
        client_sock3.close()
        self.shutdown_server_cleanly(server, server_thread)

    def test_graceful_shutdown_waits_for_clients_to_stop(self):
        server, server_thread = self.make_server()
        server.backing_transport.put_bytes('bigfile', b'a' * 1024 * 1024)
        client_sock = self.connect_to_server(server)
        self.say_hello(client_sock)
        _, server_side_thread = server._active_connections[0]
        client_medium = medium.SmartClientAlreadyConnectedSocketMedium('base', client_sock)
        client_client = client._SmartClient(client_medium)
        resp, response_handler = client_client.call_expecting_body(b'get', b'bigfile')
        self.assertEqual((b'ok',), resp)
        server._stop_gracefully()
        self.connect_to_server_and_hangup(server)
        server._stopped.wait()
        self.assertRaises(socket.error, self.connect_to_server, server)
        response_handler.read_body_bytes()
        client_sock.close()
        server_side_thread.join()
        server_thread.join()
        self.assertTrue(server._fully_stopped.is_set())
        log = self.get_log()
        self.assertThat(log, DocTestMatches("    INFO  Requested to stop gracefully\n... Stopping SmartServerSocketStreamMedium(client=('127.0.0.1', ...\n", flags=doctest.ELLIPSIS | doctest.REPORT_UDIFF))

    def test_stop_gracefully_tells_handlers_to_stop(self):
        server, server_thread = self.make_server()
        client_sock = self.connect_to_server(server)
        self.say_hello(client_sock)
        server_handler, server_side_thread = server._active_connections[0]
        self.assertFalse(server_handler.finished)
        server._stop_gracefully()
        self.assertTrue(server_handler.finished)
        client_sock.close()
        self.connect_to_server_and_hangup(server)
        server_thread.join()
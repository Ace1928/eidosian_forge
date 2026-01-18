import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestTCPServerInAThread(tests.TestCase):
    scenarios = [(name, {'server_class': getattr(test_server, name)}) for name in ('TestingTCPServer', 'TestingThreadingTCPServer')]

    def get_server(self, server_class=None, connection_handler_class=None):
        if server_class is not None:
            self.server_class = server_class
        if connection_handler_class is None:
            connection_handler_class = TCPConnectionHandler
        server = test_server.TestingTCPServerInAThread(('localhost', 0), self.server_class, connection_handler_class)
        server.start_server()
        self.addCleanup(server.stop_server)
        return server

    def get_client(self):
        client = TCPClient()
        self.addCleanup(client.disconnect)
        return client

    def get_server_connection(self, server, conn_rank):
        return server.server.clients[conn_rank]

    def assertClientAddr(self, client, server, conn_rank):
        conn = self.get_server_connection(server, conn_rank)
        self.assertEqual(client.sock.getsockname(), conn[1])

    def test_start_stop(self):
        server = self.get_server()
        client = self.get_client()
        server.stop_server()
        client = self.get_client()
        self.assertRaises(socket.error, client.connect, (server.host, server.port))

    def test_client_talks_server_respond(self):
        server = self.get_server()
        client = self.get_client()
        client.connect((server.host, server.port))
        self.assertIs(None, client.write(b'ping\n'))
        resp = client.read()
        self.assertClientAddr(client, server, 0)
        self.assertEqual(b'pong\n', resp)

    def test_server_fails_to_start(self):

        class CantStart(Exception):
            pass

        class CantStartServer(test_server.TestingTCPServer):

            def server_bind(self):
                raise CantStart()
        self.assertRaises(CantStart, self.get_server, server_class=CantStartServer)

    def test_server_fails_while_serving_or_stopping(self):

        class CantConnect(Exception):
            pass

        class FailingConnectionHandler(TCPConnectionHandler):

            def handle(self):
                raise CantConnect()
        server = self.get_server(connection_handler_class=FailingConnectionHandler)
        client = self.get_client()
        client.connect((server.host, server.port))
        client.write(b'ping\n')
        try:
            self.assertEqual(b'', client.read())
        except OSError as e:
            WSAECONNRESET = 10054
            if e.errno in (WSAECONNRESET,):
                pass
        self.assertRaises(CantConnect, server.stop_server)

    def test_server_crash_while_responding(self):
        caught = threading.Event()
        caught.clear()
        self.connection_thread = None

        class FailToRespond(Exception):
            pass

        class FailingDuringResponseHandler(TCPConnectionHandler):

            def handle_connection(request):
                request.readline()
                self.connection_thread = threading.currentThread()
                self.connection_thread.set_sync_event(caught)
                raise FailToRespond()
        server = self.get_server(connection_handler_class=FailingDuringResponseHandler)
        client = self.get_client()
        client.connect((server.host, server.port))
        client.write(b'ping\n')
        caught.wait()
        self.assertEqual(b'', client.read())
        self.assertRaises(FailToRespond, self.connection_thread.pending_exception)

    def test_exception_swallowed_while_serving(self):
        caught = threading.Event()
        caught.clear()
        self.connection_thread = None

        class CantServe(Exception):
            pass

        class FailingWhileServingConnectionHandler(TCPConnectionHandler):

            def handle(request):
                self.connection_thread = threading.currentThread()
                self.connection_thread.set_sync_event(caught)
                raise CantServe()
        server = self.get_server(connection_handler_class=FailingWhileServingConnectionHandler)
        self.assertEqual(True, server.server.serving)
        server.set_ignored_exceptions(CantServe)
        client = self.get_client()
        client.connect((server.host, server.port))
        caught.wait()
        self.assertEqual(b'', client.read())
        self.assertIs(None, self.connection_thread.pending_exception())
        self.assertIs(None, server.pending_exception())

    def test_handle_request_closes_if_it_doesnt_process(self):
        server = self.get_server()
        client = self.get_client()
        server.server.serving = False
        try:
            client.connect((server.host, server.port))
            self.assertEqual(b'', client.read())
        except OSError as e:
            if e.errno != errno.ECONNRESET:
                raise
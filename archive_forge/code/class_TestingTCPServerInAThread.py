import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
class TestingTCPServerInAThread(transport.Server):
    """A server in a thread that re-raise thread exceptions."""

    def __init__(self, server_address, server_class, request_handler_class):
        self.server_class = server_class
        self.request_handler_class = request_handler_class
        self.host, self.port = server_address
        self.server = None
        self._server_thread = None

    def __repr__(self):
        return '{}({}:{})'.format(self.__class__.__name__, self.host, self.port)

    def create_server(self):
        return self.server_class((self.host, self.port), self.request_handler_class)

    def start_server(self):
        self.server = self.create_server()
        self._server_thread = TestThread(sync_event=self.server.started, target=self.run_server)
        self._server_thread.start()
        self.server.started.wait()
        self.host, self.port = self.server.server_address
        self._server_thread.name = self.server.server_address
        if debug_threads():
            sys.stderr.write('Server thread %s started\n' % (self._server_thread.name,))
        self._server_thread.pending_exception()
        self._server_thread.set_sync_event(self.server.stopped)

    def run_server(self):
        self.server.serve()

    def stop_server(self):
        if self.server is None:
            return
        try:
            self.set_ignored_exceptions(self.server.ignored_exceptions_during_shutdown)
            self.server.serving = False
            if debug_threads():
                sys.stderr.write('Server thread %s will be joined\n' % (self._server_thread.name,))
            last_conn = None
            try:
                last_conn = osutils.connect_socket((self.host, self.port))
            except OSError:
                pass
            self.server.stop_client_connections()
            self.server.stopped.wait()
            if last_conn is not None:
                last_conn.close()
            try:
                self._server_thread.join()
            except Exception as e:
                if self.server.ignored_exceptions(e):
                    pass
                else:
                    raise
        finally:
            self.server = None

    def set_ignored_exceptions(self, ignored_exceptions):
        """Install an exception handler for the server."""
        self.server.set_ignored_exceptions(self._server_thread, ignored_exceptions)

    def pending_exception(self):
        """Raise uncaught exception in the server."""
        self.server._pending_exception(self._server_thread)
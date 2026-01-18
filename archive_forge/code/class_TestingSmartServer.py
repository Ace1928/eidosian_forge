import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
class TestingSmartServer(TestingThreadingTCPServer, server.SmartTCPServer):

    def __init__(self, server_address, request_handler_class, backing_transport, root_client_path):
        TestingThreadingTCPServer.__init__(self, server_address, request_handler_class)
        server.SmartTCPServer.__init__(self, backing_transport, root_client_path, client_timeout=_DEFAULT_TESTING_CLIENT_TIMEOUT)

    def serve(self):
        self.run_server_started_hooks()
        try:
            TestingThreadingTCPServer.serve(self)
        finally:
            self.run_server_stopped_hooks()

    def get_url(self):
        """Return the url of the server"""
        return 'bzr://%s:%d/' % self.server_address
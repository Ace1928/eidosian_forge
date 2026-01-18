import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
class TestingSmartConnectionHandler(socketserver.BaseRequestHandler, medium.SmartServerSocketStreamMedium):

    def __init__(self, request, client_address, server):
        medium.SmartServerSocketStreamMedium.__init__(self, request, server.backing_transport, server.root_client_path, timeout=_DEFAULT_TESTING_CLIENT_TIMEOUT)
        request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        socketserver.BaseRequestHandler.__init__(self, request, client_address, server)

    def handle(self):
        try:
            while not self.finished:
                server_protocol = self._build_protocol()
                self._serve_one_request(server_protocol)
        except errors.ConnectionTimeout:
            return
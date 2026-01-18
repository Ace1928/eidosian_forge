import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
class FakeSocket:
    """Fake socket object used to test the SocketDelay wrapper without
    using a real socket.
    """

    def __init__(self):
        self._data = ''

    def send(self, data, flags=0):
        self._data += data
        return len(data)

    def sendall(self, data, flags=0):
        self._data += data
        return len(data)

    def recv(self, size, flags=0):
        if size < len(self._data):
            result = self._data[:size]
            self._data = self._data[size:]
            return result
        else:
            result = self._data
            self._data = ''
            return result
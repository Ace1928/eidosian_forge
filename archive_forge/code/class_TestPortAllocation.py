import errno
import os
import signal
import socket
from subprocess import Popen
import sys
import time
import unittest
from tornado.netutil import (
from tornado.testing import AsyncTestCase, gen_test, bind_unused_port
from tornado.test.util import skipIfNoNetwork
import typing
class TestPortAllocation(unittest.TestCase):

    def test_same_port_allocation(self):
        if 'TRAVIS' in os.environ:
            self.skipTest('dual-stack servers often have port conflicts on travis')
        sockets = bind_sockets(0, 'localhost')
        try:
            port = sockets[0].getsockname()[1]
            self.assertTrue(all((s.getsockname()[1] == port for s in sockets[1:])))
        finally:
            for sock in sockets:
                sock.close()

    @unittest.skipIf(not hasattr(socket, 'SO_REUSEPORT'), 'SO_REUSEPORT is not supported')
    def test_reuse_port(self):
        sockets = []
        socket, port = bind_unused_port(reuse_port=True)
        try:
            sockets = bind_sockets(port, '127.0.0.1', reuse_port=True)
            self.assertTrue(all((s.getsockname()[1] == port for s in sockets)))
        finally:
            socket.close()
            for sock in sockets:
                sock.close()
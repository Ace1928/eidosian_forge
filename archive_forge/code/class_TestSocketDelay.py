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
class TestSocketDelay(TestCase):

    def setUp(self):
        super().setUp()
        self.requireFeature(features.paramiko)

    def test_delay(self):
        sending = FakeSocket()
        receiving = stub_sftp.SocketDelay(sending, 0.1, bandwidth=1000000, really_sleep=False)
        t1 = stub_sftp.SocketDelay.simulated_time
        receiving.send('connect1')
        self.assertEqual(sending.recv(1024), 'connect1')
        t2 = stub_sftp.SocketDelay.simulated_time
        self.assertAlmostEqual(t2 - t1, 0.1)
        receiving.send('connect2')
        self.assertEqual(sending.recv(1024), 'connect2')
        sending.send('hello')
        self.assertEqual(receiving.recv(1024), 'hello')
        t3 = stub_sftp.SocketDelay.simulated_time
        self.assertAlmostEqual(t3 - t2, 0.1)
        sending.send('hello')
        self.assertEqual(receiving.recv(1024), 'hello')
        sending.send('hello')
        self.assertEqual(receiving.recv(1024), 'hello')
        sending.send('hello')
        self.assertEqual(receiving.recv(1024), 'hello')
        t4 = stub_sftp.SocketDelay.simulated_time
        self.assertAlmostEqual(t4, t3)

    def test_bandwidth(self):
        sending = FakeSocket()
        receiving = stub_sftp.SocketDelay(sending, 0, bandwidth=8.0 / (1024 * 1024), really_sleep=False)
        t1 = stub_sftp.SocketDelay.simulated_time
        receiving.send('connect')
        self.assertEqual(sending.recv(1024), 'connect')
        sending.send('a' * 100)
        self.assertEqual(receiving.recv(1024), 'a' * 100)
        t2 = stub_sftp.SocketDelay.simulated_time
        self.assertAlmostEqual(t2 - t1, 100 + 7)
import os
import re
import struct
from unittest import skipIf
from hamcrest import assert_that, equal_to
from twisted.internet import defer
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import StringTransport
from twisted.protocols import loopback
from twisted.python import components
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
@skipIf(not unix, "can't run on non-posix computers")
class FileTransferCloseTests(TestCase):

    def setUp(self):
        self.avatar = TestAvatar()

    def buildServerConnection(self):
        conn = connection.SSHConnection()

        class DummyTransport:

            def __init__(self):
                self.transport = self

            def sendPacket(self, kind, data):
                pass

            def logPrefix(self):
                return 'dummy transport'
        conn.transport = DummyTransport()
        conn.transport.avatar = self.avatar
        return conn

    def interceptConnectionLost(self, sftpServer):
        self.connectionLostFired = False
        origConnectionLost = sftpServer.connectionLost

        def connectionLost(reason):
            self.connectionLostFired = True
            origConnectionLost(reason)
        sftpServer.connectionLost = connectionLost

    def assertSFTPConnectionLost(self):
        self.assertTrue(self.connectionLostFired, "sftpServer's connectionLost was not called")

    def test_sessionClose(self):
        """
        Closing a session should notify an SFTP subsystem launched by that
        session.
        """
        testSession = session.SSHSession(conn=FakeConn(), avatar=self.avatar)
        testSession.request_subsystem(common.NS(b'sftp'))
        sftpServer = testSession.client.transport.proto
        self.interceptConnectionLost(sftpServer)
        testSession.closeReceived()
        self.assertSFTPConnectionLost()

    def test_clientClosesChannelOnConnnection(self):
        """
        A client sending CHANNEL_CLOSE should trigger closeReceived on the
        associated channel instance.
        """
        conn = self.buildServerConnection()
        packet = common.NS(b'session') + struct.pack('>L', 0) * 3
        conn.ssh_CHANNEL_OPEN(packet)
        sessionChannel = conn.channels[0]
        sessionChannel.request_subsystem(common.NS(b'sftp'))
        sftpServer = sessionChannel.client.transport.proto
        self.interceptConnectionLost(sftpServer)
        self.interceptConnectionLost(sftpServer)
        conn.ssh_CHANNEL_CLOSE(struct.pack('>L', 0))
        self.assertSFTPConnectionLost()

    def test_stopConnectionServiceClosesChannel(self):
        """
        Closing an SSH connection should close all sessions within it.
        """
        conn = self.buildServerConnection()
        packet = common.NS(b'session') + struct.pack('>L', 0) * 3
        conn.ssh_CHANNEL_OPEN(packet)
        sessionChannel = conn.channels[0]
        sessionChannel.request_subsystem(common.NS(b'sftp'))
        sftpServer = sessionChannel.client.transport.proto
        self.interceptConnectionLost(sftpServer)
        conn.serviceStopped()
        self.assertSFTPConnectionLost()
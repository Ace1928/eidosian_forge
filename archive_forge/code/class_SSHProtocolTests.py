import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
class SSHProtocolTests(unittest.TestCase):
    """
    Tests for communication between L{SSHServerTransport} and
    L{SSHClientTransport}.
    """
    if not cryptography:
        skip = "can't run without cryptography"

    def _ourServerOurClientTest(self, name=b'session', **kwargs):
        """
        Create a connected SSH client and server protocol pair and return a
        L{Deferred} which fires with an L{SSHTestChannel} instance connected to
        a channel on that SSH connection.
        """
        result = defer.Deferred()
        self.realm = ConchTestRealm(b'testuser')
        p = portal.Portal(self.realm)
        sshpc = ConchTestSSHChecker()
        sshpc.registerChecker(ConchTestPasswordChecker())
        sshpc.registerChecker(conchTestPublicKeyChecker())
        p.registerChecker(sshpc)
        fac = ConchTestServerFactory()
        fac.portal = p
        fac.startFactory()
        self.server = fac.buildProtocol(None)
        self.clientTransport = LoopbackRelay(self.server)
        self.client = ConchTestClient(lambda conn: SSHTestChannel(name, result, conn=conn, **kwargs))
        self.serverTransport = LoopbackRelay(self.client)
        self.server.makeConnection(self.serverTransport)
        self.client.makeConnection(self.clientTransport)
        return result

    def test_subsystemsAndGlobalRequests(self):
        """
        Run the Conch server against the Conch client.  Set up several different
        channels which exercise different behaviors and wait for them to
        complete.  Verify that the channels with errors log them.
        """
        channel = self._ourServerOurClientTest()

        def cbSubsystem(channel):
            self.channel = channel
            return self.assertFailure(channel.conn.sendRequest(channel, b'subsystem', common.NS(b'not-crazy'), 1), Exception)
        channel.addCallback(cbSubsystem)

        def cbNotCrazyFailed(ignored):
            channel = self.channel
            return channel.conn.sendRequest(channel, b'subsystem', common.NS(b'crazy'), 1)
        channel.addCallback(cbNotCrazyFailed)

        def cbGlobalRequests(ignored):
            channel = self.channel
            d1 = channel.conn.sendGlobalRequest(b'foo', b'bar', 1)
            d2 = channel.conn.sendGlobalRequest(b'foo-2', b'bar2', 1)
            d2.addCallback(self.assertEqual, b'data')
            d3 = self.assertFailure(channel.conn.sendGlobalRequest(b'bar', b'foo', 1), Exception)
            return defer.gatherResults([d1, d2, d3])
        channel.addCallback(cbGlobalRequests)

        def disconnect(ignored):
            self.assertEqual(self.realm.avatar.globalRequests, {'foo': b'bar', 'foo_2': b'bar2'})
            channel = self.channel
            channel.conn.transport.expectedLoseConnection = True
            channel.conn.serviceStopped()
            channel.loseConnection()
        channel.addCallback(disconnect)
        return channel

    def test_shell(self):
        """
        L{SSHChannel.sendRequest} can open a shell with a I{pty-req} request,
        specifying a terminal type and window size.
        """
        channel = self._ourServerOurClientTest()
        data = session.packRequest_pty_req(b'conch-test-term', (24, 80, 0, 0), b'')

        def cbChannel(channel):
            self.channel = channel
            return channel.conn.sendRequest(channel, b'pty-req', data, 1)
        channel.addCallback(cbChannel)

        def cbPty(ignored):
            session = self.realm.avatar.conn.channels[0].session
            self.assertIs(session.avatar, self.realm.avatar)
            self.assertEqual(session._terminalType, b'conch-test-term')
            self.assertEqual(session._windowSize, (24, 80, 0, 0))
            self.assertTrue(session.ptyReq)
            channel = self.channel
            return channel.conn.sendRequest(channel, b'shell', b'', 1)
        channel.addCallback(cbPty)

        def cbShell(ignored):
            self.channel.write(b'testing the shell!\x00')
            self.channel.conn.sendEOF(self.channel)
            return defer.gatherResults([self.channel.onClose, self.realm.avatar._testSession.onClose])
        channel.addCallback(cbShell)

        def cbExited(ignored):
            if self.channel.status != 0:
                log.msg('shell exit status was not 0: %i' % (self.channel.status,))
            self.assertEqual(b''.join(self.channel.received), b'testing the shell!\x00\r\n')
            self.assertTrue(self.channel.eofCalled)
            self.assertTrue(self.realm.avatar._testSession.eof)
        channel.addCallback(cbExited)
        return channel

    def test_failedExec(self):
        """
        If L{SSHChannel.sendRequest} issues an exec which the server responds to
        with an error, the L{Deferred} it returns fires its errback.
        """
        channel = self._ourServerOurClientTest()

        def cbChannel(channel):
            self.channel = channel
            return self.assertFailure(channel.conn.sendRequest(channel, b'exec', common.NS(b'jumboliah'), 1), Exception)
        channel.addCallback(cbChannel)

        def cbFailed(ignored):
            errors = self.flushLoggedErrors(error.ConchError)
            self.assertEqual(errors[0].value.args, ('bad exec', None))
        channel.addCallback(cbFailed)
        return channel

    def test_falseChannel(self):
        """
        When the process started by a L{SSHChannel.sendRequest} exec request
        exits, the exit status is reported to the channel.
        """
        channel = self._ourServerOurClientTest()

        def cbChannel(channel):
            self.channel = channel
            return channel.conn.sendRequest(channel, b'exec', common.NS(b'false'), 1)
        channel.addCallback(cbChannel)

        def cbExec(ignored):
            return self.channel.onClose
        channel.addCallback(cbExec)

        def cbClosed(ignored):
            self.assertEqual(self.channel.received, [])
            self.assertNotEqual(self.channel.status, 0)
        channel.addCallback(cbClosed)
        return channel

    def test_errorChannel(self):
        """
        Bytes sent over the extended channel for stderr data are delivered to
        the channel's C{extReceived} method.
        """
        channel = self._ourServerOurClientTest(localWindow=4, localMaxPacket=5)

        def cbChannel(channel):
            self.channel = channel
            return channel.conn.sendRequest(channel, b'exec', common.NS(b'eecho hello'), 1)
        channel.addCallback(cbChannel)

        def cbExec(ignored):
            return defer.gatherResults([self.channel.onClose, self.realm.avatar._testSession.onClose])
        channel.addCallback(cbExec)

        def cbClosed(ignored):
            self.assertEqual(self.channel.received, [])
            self.assertEqual(b''.join(self.channel.receivedExt), b'hello\r\n')
            self.assertEqual(self.channel.status, 0)
            self.assertTrue(self.channel.eofCalled)
            self.assertEqual(self.channel.localWindowLeft, 4)
            self.assertEqual(self.channel.localWindowLeft, self.realm.avatar._testSession.remoteWindowLeftAtClose)
        channel.addCallback(cbClosed)
        return channel

    def test_unknownChannel(self):
        """
        When an attempt is made to open an unknown channel type, the L{Deferred}
        returned by L{SSHChannel.sendRequest} fires its errback.
        """
        d = self.assertFailure(self._ourServerOurClientTest(b'crazy-unknown-channel'), Exception)

        def cbFailed(ignored):
            errors = self.flushLoggedErrors(error.ConchError)
            self.assertEqual(errors[0].value.args, (3, 'unknown channel'))
            self.assertEqual(len(errors), 1)
        d.addCallback(cbFailed)
        return d

    def test_maxPacket(self):
        """
        An L{SSHChannel} can be configured with a maximum packet size to
        receive.
        """
        channel = self._ourServerOurClientTest(localWindow=11, localMaxPacket=1)

        def cbChannel(channel):
            self.channel = channel
            return channel.conn.sendRequest(channel, b'exec', common.NS(b'secho hello'), 1)
        channel.addCallback(cbChannel)

        def cbExec(ignored):
            return self.channel.onClose
        channel.addCallback(cbExec)

        def cbClosed(ignored):
            self.assertEqual(self.channel.status, 0)
            self.assertEqual(b''.join(self.channel.received), b'hello\r\n')
            self.assertEqual(b''.join(self.channel.receivedExt), b'hello\r\n')
            self.assertEqual(self.channel.localWindowLeft, 11)
            self.assertTrue(self.channel.eofCalled)
        channel.addCallback(cbClosed)
        return channel

    def test_echo(self):
        """
        Normal standard out bytes are sent to the channel's C{dataReceived}
        method.
        """
        channel = self._ourServerOurClientTest(localWindow=4, localMaxPacket=5)

        def cbChannel(channel):
            self.channel = channel
            return channel.conn.sendRequest(channel, b'exec', common.NS(b'echo hello'), 1)
        channel.addCallback(cbChannel)

        def cbEcho(ignored):
            return defer.gatherResults([self.channel.onClose, self.realm.avatar._testSession.onClose])
        channel.addCallback(cbEcho)

        def cbClosed(ignored):
            self.assertEqual(self.channel.status, 0)
            self.assertEqual(b''.join(self.channel.received), b'hello\r\n')
            self.assertEqual(self.channel.localWindowLeft, 4)
            self.assertTrue(self.channel.eofCalled)
            self.assertEqual(self.channel.localWindowLeft, self.realm.avatar._testSession.remoteWindowLeftAtClose)
        channel.addCallback(cbClosed)
        return channel
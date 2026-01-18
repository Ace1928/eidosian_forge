import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
class SessionInterfaceTests(RegistryUsingMixin, TestCase):
    """
    Tests for the SSHSession class interface.  This interface is not ideal, but
    it is tested in order to maintain backwards compatibility.
    """
    if not cryptography:
        skip = 'cannot run without cryptography'

    def setUp(self, register_adapters=True):
        """
        Make an SSHSession object to test.  Give the channel some window
        so that it's allowed to send packets.  500 and 100 are arbitrary
        values.
        """
        RegistryUsingMixin.setUp(self)
        self.session = self.getSSHSession()
        if register_adapters:
            components.registerAdapter(StubSessionForStubAvatarWithEnv, StubAvatar, session.ISession)
        self.session = self.getSSHSession()

    def getSSHSession(self, register_adapters=True):
        """
        Return a new SSH session.
        """
        return session.SSHSession(remoteWindow=500, remoteMaxPacket=100, conn=StubConnection(), avatar=StubAvatar())

    def assertSessionIsStubSession(self):
        """
        Asserts that self.session.session is an instance of
        StubSessionForStubOldAvatar.
        """
        self.assertIsInstance(self.session.session, StubSessionForStubAvatar)

    def test_init(self):
        """
        SSHSession initializes its buffer (buf), client, and ISession adapter.
        The avatar should not need to be adaptable to an ISession immediately.
        """
        s = session.SSHSession(avatar=object)
        self.assertEqual(s.buf, b'')
        self.assertIsNone(s.client)
        self.assertIsNone(s.session)

    def test_client_dataReceived(self):
        """
        SSHSession.dataReceived() passes data along to a client.  If the data
        comes before there is a client, the data should be discarded.
        """
        self.session.dataReceived(b'1')
        self.session.client = StubClient()
        self.session.dataReceived(b'2')
        self.assertEqual(self.session.client.transport.buf, b'2')

    def test_client_extReceived(self):
        """
        SSHSession.extReceived() passed data of type EXTENDED_DATA_STDERR along
        to the client.  If the data comes before there is a client, or if the
        data is not of type EXTENDED_DATA_STDERR, it is discared.
        """
        self.session.extReceived(connection.EXTENDED_DATA_STDERR, b'1')
        self.session.extReceived(255, b'2')
        self.session.client = StubClient()
        self.session.extReceived(connection.EXTENDED_DATA_STDERR, b'3')
        self.assertEqual(self.session.client.transport.err, b'3')

    def test_client_extReceivedWithoutWriteErr(self):
        """
        SSHSession.extReceived() should handle the case where the transport
        on the client doesn't have a writeErr method.
        """
        client = self.session.client = StubClient()
        client.transport = StubTransport()
        self.session.extReceived(connection.EXTENDED_DATA_STDERR, b'ignored')

    def test_client_closed(self):
        """
        SSHSession.closed() should tell the transport connected to the client
        that the connection was lost.
        """
        self.session.client = StubClient()
        self.session.closed()
        self.assertTrue(self.session.client.transport.close)
        self.session.client.transport.close = False

    def test_client_closed_with_env_subsystem(self):
        """
        If the peer requests an environment variable in its setup process
        followed by requesting a subsystem, SSHSession.closed() should tell
        the transport connected to the client that the connection was lost.
        """
        self.assertTrue(self.session.requestReceived(b'env', common.NS(b'FOO') + common.NS(b'bar')))
        self.assertTrue(self.session.requestReceived(b'subsystem', common.NS(b'TestSubsystem') + b'data'))
        self.session.client = StubClient()
        self.session.closed()
        self.assertTrue(self.session.client.transport.close)
        self.session.client.transport.close = False

    def test_badSubsystemDoesNotCreateClient(self):
        """
        When a subsystem request fails, SSHSession.client should not be set.
        """
        ret = self.session.requestReceived(b'subsystem', common.NS(b'BadSubsystem'))
        self.assertFalse(ret)
        self.assertIsNone(self.session.client)

    def test_lookupSubsystem(self):
        """
        When a client requests a subsystem, the SSHSession object should get
        the subsystem by calling avatar.lookupSubsystem, and attach it as
        the client.
        """
        ret = self.session.requestReceived(b'subsystem', common.NS(b'TestSubsystem') + b'data')
        self.assertTrue(ret)
        self.assertIsInstance(self.session.client, protocol.ProcessProtocol)
        self.assertIs(self.session.client.transport.proto, self.session.avatar.subsystem)

    def test_lookupSubsystemDoesNotNeedISession(self):
        """
        Previously, if one only wanted to implement a subsystem, an ISession
        adapter wasn't needed because subsystems were looked up using the
        lookupSubsystem method on the avatar.
        """
        s = session.SSHSession(avatar=SubsystemOnlyAvatar(), conn=StubConnection())
        ret = s.request_subsystem(common.NS(b'subsystem') + b'data')
        self.assertTrue(ret)
        self.assertIsNotNone(s.client)
        self.assertIsNone(s.conn.closes.get(s))
        s.eofReceived()
        self.assertTrue(s.conn.closes.get(s))
        s.loseConnection()
        s.closed()

    def test_lookupSubsystem_data(self):
        """
        After having looked up a subsystem, data should be passed along to the
        client.  Additionally, subsystems were passed the entire request packet
        as data, instead of just the additional data.

        We check for the additional tidle to verify that the data passed
        through the client.
        """
        self.session.requestReceived(b'subsystem', common.NS(b'TestSubsystem') + b'data')
        self.assertEqual(self.session.conn.data[self.session], [b'\x00\x00\x00\rTestSubsystemdata~'])
        self.session.dataReceived(b'more data')
        self.assertEqual(self.session.conn.data[self.session][-1], b'more data~')

    def test_lookupSubsystem_closeReceived(self):
        """
        SSHSession.closeReceived() should sent a close message to the remote
        side.
        """
        self.session.requestReceived(b'subsystem', common.NS(b'TestSubsystem') + b'data')
        self.session.closeReceived()
        self.assertTrue(self.session.conn.closes[self.session])

    def assertRequestRaisedRuntimeError(self):
        """
        Assert that the request we just made raised a RuntimeError (and only a
        RuntimeError).
        """
        errors = self.flushLoggedErrors(RuntimeError)
        self.assertEqual(len(errors), 1, 'Multiple RuntimeErrors raised: %s' % '\n'.join([repr(error) for error in errors]))
        errors[0].trap(RuntimeError)

    def test_requestShell(self):
        """
        When a client requests a shell, the SSHSession object should get
        the shell by getting an ISession adapter for the avatar, then
        calling openShell() with a ProcessProtocol to attach.
        """
        ret = self.session.requestReceived(b'shell', b'')
        self.assertTrue(ret)
        self.assertSessionIsStubSession()
        self.assertIsInstance(self.session.client, session.SSHSessionProcessProtocol)
        self.assertIs(self.session.session.shellProtocol, self.session.client)
        self.assertFalse(self.session.requestReceived(b'shell', b''))
        self.assertRequestRaisedRuntimeError()

    def test_requestShellWithData(self):
        """
        When a client executes a shell, it should be able to give pass data
        back and forth between the local and the remote side.
        """
        ret = self.session.requestReceived(b'shell', b'')
        self.assertTrue(ret)
        self.assertSessionIsStubSession()
        self.session.dataReceived(b'some data\x00')
        self.assertEqual(self.session.session.shellTransport.data, b'some data\x00')
        self.assertEqual(self.session.conn.data[self.session], [b'some data\x00', b'\r\n'])
        self.assertTrue(self.session.session.shellTransport.closed)
        self.assertEqual(self.session.conn.requests[self.session], [(b'exit-status', b'\x00\x00\x00\x00', False)])

    def test_requestExec(self):
        """
        When a client requests a command, the SSHSession object should get
        the command by getting an ISession adapter for the avatar, then
        calling execCommand with a ProcessProtocol to attach and the
        command line.
        """
        ret = self.session.requestReceived(b'exec', common.NS(b'failure'))
        self.assertFalse(ret)
        self.assertRequestRaisedRuntimeError()
        self.assertIsNone(self.session.client)
        self.assertTrue(self.session.requestReceived(b'exec', common.NS(b'success')))
        self.assertSessionIsStubSession()
        self.assertIsInstance(self.session.client, session.SSHSessionProcessProtocol)
        self.assertIs(self.session.session.execProtocol, self.session.client)
        self.assertEqual(self.session.session.execCommandLine, b'success')

    def test_requestExecWithData(self):
        """
        When a client executes a command, it should be able to give pass data
        back and forth.
        """
        ret = self.session.requestReceived(b'exec', common.NS(b'repeat hello'))
        self.assertTrue(ret)
        self.assertSessionIsStubSession()
        self.session.dataReceived(b'some data')
        self.assertEqual(self.session.session.execTransport.data, b'some data')
        self.assertEqual(self.session.conn.data[self.session], [b'hello', b'some data', b'\r\n'])
        self.session.eofReceived()
        self.session.closeReceived()
        self.session.closed()
        self.assertTrue(self.session.session.execTransport.closed)
        self.assertEqual(self.session.conn.requests[self.session], [(b'exit-status', b'\x00\x00\x00\x00', False)])

    def test_requestPty(self):
        """
        When a client requests a PTY, the SSHSession object should make
        the request by getting an ISession adapter for the avatar, then
        calling getPty with the terminal type, the window size, and any modes
        the client gave us.
        """
        self.doCleanups()
        self.setUp(register_adapters=False)
        components.registerAdapter(StubSessionForStubAvatar, StubAvatar, session.ISession)
        test_session = self.getSSHSession()
        ret = test_session.requestReceived(b'pty_req', session.packRequest_pty_req(b'bad', (1, 2, 3, 4), b''))
        self.assertFalse(ret)
        self.assertIsInstance(test_session.session, StubSessionForStubAvatar)
        self.assertRequestRaisedRuntimeError()
        self.assertTrue(test_session.requestReceived(b'pty_req', session.packRequest_pty_req(b'good', (1, 2, 3, 4), b'')))
        self.assertEqual(test_session.session.ptyRequest, (b'good', (1, 2, 3, 4), []))

    def test_setEnv(self):
        """
        When a client requests passing an environment variable, the
        SSHSession object should make the request by getting an
        ISessionSetEnv adapter for the avatar, then calling setEnv with the
        environment variable name and value.
        """
        self.assertFalse(self.session.requestReceived(b'env', common.NS(b'FAIL') + common.NS(b'bad')))
        self.assertIsInstance(self.session.session, StubSessionForStubAvatarWithEnv)
        self.assertRequestRaisedRuntimeError()
        self.assertFalse(self.session.requestReceived(b'env', common.NS(b'IGNORED') + common.NS(b'ignored')))
        self.assertEqual(self.flushLoggedErrors(), [])
        self.assertTrue(self.session.requestReceived(b'env', common.NS(b'NAME') + common.NS(b'value')))
        self.assertEqual(self.session.session.environ, {b'NAME': b'value'})

    def test_setEnvSessionShare(self):
        """
        Multiple setenv requests will share the same session.
        """
        test_session = self.getSSHSession()
        self.assertTrue(test_session.requestReceived(b'env', common.NS(b'Key1') + common.NS(b'Value 1')))
        self.assertTrue(test_session.requestReceived(b'env', common.NS(b'Key2') + common.NS(b'Value2')))
        self.assertIsInstance(test_session.session, StubSessionForStubAvatarWithEnv)
        self.assertEqual({b'Key1': b'Value 1', b'Key2': b'Value2'}, test_session.session.environ)

    def test_setEnvMultiplexShare(self):
        """
        Calling another session service after setenv will provide the
        previous session with the environment variables.
        """
        test_session = self.getSSHSession()
        test_session.requestReceived(b'env', common.NS(b'Key1') + common.NS(b'Value 1'))
        test_session.requestReceived(b'env', common.NS(b'Key2') + common.NS(b'Value2'))
        test_session.requestReceived(b'pty_req', session.packRequest_pty_req(b'term', (0, 0, 0, 0), b''))
        self.assertIsInstance(test_session.session, StubSessionForStubAvatarWithEnv)
        self.assertEqual({b'Key1': b'Value 1', b'Key2': b'Value2'}, test_session.session.environAtPty)

    def test_setEnvNotProvidingISessionSetEnv(self):
        """
        If the avatar does not have an ISessionSetEnv adapter, then a
        request to pass an environment variable fails gracefully.
        """
        self.doCleanups()
        self.setUp(register_adapters=False)
        components.registerAdapter(StubSessionForStubAvatar, StubAvatar, session.ISession)
        self.assertFalse(self.session.requestReceived(b'env', common.NS(b'NAME') + common.NS(b'value')))

    def test_requestWindowChange(self):
        """
        When the client requests to change the window size, the SSHSession
        object should make the request by getting an ISession adapter for the
        avatar, then calling windowChanged with the new window size.
        """
        ret = self.session.requestReceived(b'window_change', session.packRequest_window_change((0, 0, 0, 0)))
        self.assertFalse(ret)
        self.assertRequestRaisedRuntimeError()
        self.assertSessionIsStubSession()
        self.assertTrue(self.session.requestReceived(b'window_change', session.packRequest_window_change((1, 2, 3, 4))))
        self.assertEqual(self.session.session.windowChange, (1, 2, 3, 4))

    def test_eofReceived(self):
        """
        When an EOF is received and an ISession adapter is present, it should
        be notified of the EOF message.
        """
        self.session.session = session.ISession(self.session.avatar)
        self.session.eofReceived()
        self.assertTrue(self.session.session.gotEOF)

    def test_closeReceived(self):
        """
        When a close is received, the session should send a close message.
        """
        ret = self.session.closeReceived()
        self.assertIsNone(ret)
        self.assertTrue(self.session.conn.closes[self.session])

    def test_closed(self):
        """
        When a close is received and an ISession adapter is present, it should
        be notified of the close message.
        """
        self.session.session = session.ISession(self.session.avatar)
        self.session.closed()
        self.assertTrue(self.session.session.gotClosed)
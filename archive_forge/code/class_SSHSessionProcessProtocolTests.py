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
class SSHSessionProcessProtocolTests(TestCase):
    """
    Tests for L{SSHSessionProcessProtocol}.
    """
    if not cryptography:
        skip = 'cannot run without cryptography'

    def setUp(self):
        self.transport = StubTransport()
        self.session = session.SSHSession(conn=StubConnection(self.transport), remoteWindow=500, remoteMaxPacket=100)
        self.pp = session.SSHSessionProcessProtocol(self.session)
        self.pp.makeConnection(self.transport)

    def assertSessionClosed(self):
        """
        Assert that C{self.session} is closed.
        """
        self.assertTrue(self.session.conn.closes[self.session])

    def assertRequestsEqual(self, expectedRequests):
        """
        Assert that C{self.session} has sent the C{expectedRequests}.
        """
        self.assertEqual(self.session.conn.requests[self.session], expectedRequests)

    def test_init(self):
        """
        SSHSessionProcessProtocol should set self.session to the session passed
        to the __init__ method.
        """
        self.assertEqual(self.pp.session, self.session)

    def test_getHost(self):
        """
        SSHSessionProcessProtocol.getHost() just delegates to its
        session.conn.transport.getHost().
        """
        self.assertEqual(self.session.conn.transport.getHost(), self.pp.getHost())

    def test_getPeer(self):
        """
        SSHSessionProcessProtocol.getPeer() just delegates to its
        session.conn.transport.getPeer().
        """
        self.assertEqual(self.session.conn.transport.getPeer(), self.pp.getPeer())

    def test_connectionMade(self):
        """
        SSHSessionProcessProtocol.connectionMade() should check if there's a
        'buf' attribute on its session and write it to the transport if so.
        """
        self.session.buf = b'buffer'
        self.pp.connectionMade()
        self.assertEqual(self.transport.buf, b'buffer')

    @skipIf(not hasattr(signal, 'SIGALRM'), 'Not all signals available')
    def test_getSignalName(self):
        """
        _getSignalName should return the name of a signal when given the
        signal number.
        """
        for signalName in session.SUPPORTED_SIGNALS:
            signalName = 'SIG' + signalName
            signalValue = getattr(signal, signalName)
            sshName = self.pp._getSignalName(signalValue)
            self.assertEqual(sshName, signalName, '%i: %s != %s' % (signalValue, sshName, signalName))

    @skipIf(not hasattr(signal, 'SIGALRM'), 'Not all signals available')
    def test_getSignalNameWithLocalSignal(self):
        """
        If there are signals in the signal module which aren't in the SSH RFC,
        we map their name to [signal name]@[platform].
        """
        signal.SIGTwistedTest = signal.NSIG + 1
        self.pp._signalValuesToNames = None
        self.assertEqual(self.pp._getSignalName(signal.SIGTwistedTest), 'SIGTwistedTest@' + sys.platform)

    def test_outReceived(self):
        """
        When data is passed to the outReceived method, it should be sent to
        the session's write method.
        """
        self.pp.outReceived(b'test data')
        self.assertEqual(self.session.conn.data[self.session], [b'test data'])

    def test_write(self):
        """
        When data is passed to the write method, it should be sent to the
        session channel's write method.
        """
        self.pp.write(b'test data')
        self.assertEqual(self.session.conn.data[self.session], [b'test data'])

    def test_writeSequence(self):
        """
        When a sequence is passed to the writeSequence method, it should be
        joined together and sent to the session channel's write method.
        """
        self.pp.writeSequence([b'test ', b'data'])
        self.assertEqual(self.session.conn.data[self.session], [b'test data'])

    def test_errReceived(self):
        """
        When data is passed to the errReceived method, it should be sent to
        the session's writeExtended method.
        """
        self.pp.errReceived(b'test data')
        self.assertEqual(self.session.conn.extData[self.session], [(1, b'test data')])

    def test_outConnectionLost(self):
        """
        When outConnectionLost and errConnectionLost are both called, we should
        send an EOF message.
        """
        self.pp.outConnectionLost()
        self.assertFalse(self.session in self.session.conn.eofs)
        self.pp.errConnectionLost()
        self.assertTrue(self.session.conn.eofs[self.session])

    def test_errConnectionLost(self):
        """
        Make sure reverse ordering of events in test_outConnectionLost also
        sends EOF.
        """
        self.pp.errConnectionLost()
        self.assertFalse(self.session in self.session.conn.eofs)
        self.pp.outConnectionLost()
        self.assertTrue(self.session.conn.eofs[self.session])

    def test_loseConnection(self):
        """
        When loseConnection() is called, it should call loseConnection
        on the session channel.
        """
        self.pp.loseConnection()
        self.assertTrue(self.session.conn.closes[self.session])

    def test_connectionLost(self):
        """
        When connectionLost() is called, it should call loseConnection()
        on the session channel.
        """
        self.pp.connectionLost(failure.Failure(ProcessDone(0)))

    def test_processEndedWithExitCode(self):
        """
        When processEnded is called, if there is an exit code in the reason
        it should be sent in an exit-status method.  The connection should be
        closed.
        """
        self.pp.processEnded(Failure(ProcessDone(None)))
        self.assertRequestsEqual([(b'exit-status', struct.pack('>I', 0), False)])
        self.assertSessionClosed()

    @skipIf(not hasattr(os, 'WCOREDUMP'), "can't run this w/o os.WCOREDUMP")
    def test_processEndedWithExitSignalCoreDump(self):
        """
        When processEnded is called, if there is an exit signal in the reason
        it should be sent in an exit-signal message.  The connection should be
        closed.
        """
        self.pp.processEnded(Failure(ProcessTerminated(1, signal.SIGTERM, 1 << 7)))
        self.assertRequestsEqual([(b'exit-signal', common.NS(b'TERM') + b'\x01' + common.NS(b'') + common.NS(b''), False)])
        self.assertSessionClosed()

    @skipIf(not hasattr(os, 'WCOREDUMP'), "can't run this w/o os.WCOREDUMP")
    def test_processEndedWithExitSignalNoCoreDump(self):
        """
        When processEnded is called, if there is an exit signal in the
        reason it should be sent in an exit-signal message.  If no
        core was dumped, don't set the core-dump bit.
        """
        self.pp.processEnded(Failure(ProcessTerminated(1, signal.SIGTERM, 0)))
        self.assertRequestsEqual([(b'exit-signal', common.NS(b'TERM') + b'\x00' + common.NS(b'') + common.NS(b''), False)])
        self.assertSessionClosed()
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
class WrappersTests(TestCase):
    """
    A test for the wrapProtocol and wrapProcessProtocol functions.
    """
    if not cryptography:
        skip = 'cannot run without cryptography'

    def test_wrapProtocol(self):
        """
        L{wrapProtocol}, when passed a L{Protocol} should return something that
        has write(), writeSequence(), loseConnection() methods which call the
        Protocol's dataReceived() and connectionLost() methods, respectively.
        """
        protocol = MockProtocol()
        protocol.transport = StubTransport()
        protocol.connectionMade()
        wrapped = session.wrapProtocol(protocol)
        wrapped.dataReceived(b'dataReceived')
        self.assertEqual(protocol.transport.buf, b'dataReceived')
        wrapped.write(b'data')
        wrapped.writeSequence([b'1', b'2'])
        wrapped.loseConnection()
        self.assertEqual(protocol.data, b'data12')
        protocol.reason.trap(error.ConnectionDone)

    def test_wrapProcessProtocol_Protocol(self):
        """
        L{wrapPRocessProtocol}, when passed a L{Protocol} should return
        something that follows the L{IProcessProtocol} interface, with
        connectionMade() mapping to connectionMade(), outReceived() mapping to
        dataReceived() and processEnded() mapping to connectionLost().
        """
        protocol = MockProtocol()
        protocol.transport = StubTransport()
        process_protocol = session.wrapProcessProtocol(protocol)
        process_protocol.connectionMade()
        process_protocol.outReceived(b'data')
        self.assertEqual(protocol.transport.buf, b'data~')
        process_protocol.processEnded(failure.Failure(error.ProcessTerminated(0, None, None)))
        protocol.reason.trap(error.ProcessTerminated)
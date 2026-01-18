import errno
import gc
import io
import os
import socket
from functools import wraps
from typing import Callable, ClassVar, List, Mapping, Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass, verifyObject
import attr
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import (
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import (
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.test.test_tcp import (
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
class AbortConnectionMixin:
    """
    Unit tests for L{ITransport.abortConnection}.
    """
    endpoints: Optional[EndpointCreator] = None

    def runAbortTest(self, clientClass, serverClass, clientConnectionLostReason=None):
        """
        A test runner utility function, which hooks up a matched pair of client
        and server protocols.

        We then run the reactor until both sides have disconnected, and then
        verify that the right exception resulted.
        """
        clientExpectedExceptions = (ConnectionAborted, ConnectionLost)
        serverExpectedExceptions = (ConnectionLost, ConnectionDone)
        if useSSL:
            clientExpectedExceptions = clientExpectedExceptions + (SSL.Error,)
            serverExpectedExceptions = serverExpectedExceptions + (SSL.Error,)
        client = clientClass()
        server = serverClass()
        client.otherProtocol = server
        server.otherProtocol = client
        reactor = runProtocolsWithReactor(self, server, client, self.endpoints)
        self.assertEqual(reactor.removeAll(), [])
        self.assertEqual(reactor.getDelayedCalls(), [])
        if clientConnectionLostReason is not None:
            self.assertIsInstance(client.disconnectReason.value, (clientConnectionLostReason,) + clientExpectedExceptions)
        else:
            self.assertIsInstance(client.disconnectReason.value, clientExpectedExceptions)
        self.assertIsInstance(server.disconnectReason.value, serverExpectedExceptions)

    def test_dataReceivedAbort(self):
        """
        abortConnection() is called in dataReceived. The protocol should be
        disconnected, but connectionLost should not be called re-entrantly.
        """
        return self.runAbortTest(AbortingClient, ReadAbortServerProtocol)

    def test_clientAbortsConnectionTwice(self):
        """
        abortConnection() is called twice by client.

        No exception should be thrown, and the connection will be closed.
        """
        return self.runAbortTest(AbortingTwiceClient, ReadAbortServerProtocol)

    def test_clientAbortsConnectionThenLosesConnection(self):
        """
        Client calls abortConnection(), followed by loseConnection().

        No exception should be thrown, and the connection will be closed.
        """
        return self.runAbortTest(AbortingThenLosingClient, ReadAbortServerProtocol)

    def test_serverAbortsConnectionTwice(self):
        """
        abortConnection() is called twice by server.

        No exception should be thrown, and the connection will be closed.
        """
        return self.runAbortTest(WritingButNotAbortingClient, ServerAbortsTwice, clientConnectionLostReason=ConnectionLost)

    def test_serverAbortsConnectionThenLosesConnection(self):
        """
        Server calls abortConnection(), followed by loseConnection().

        No exception should be thrown, and the connection will be closed.
        """
        return self.runAbortTest(WritingButNotAbortingClient, ServerAbortsThenLoses, clientConnectionLostReason=ConnectionLost)

    @skipIf(os.environ.get('CI', '').lower() == 'true' and platform.isMacOSX(), 'Flaky on macOS on Azure.')
    def test_resumeProducingAbort(self):
        """
        abortConnection() is called in resumeProducing, before any bytes have
        been exchanged. The protocol should be disconnected, but
        connectionLost should not be called re-entrantly.
        """
        self.runAbortTest(ProducerAbortingClient, ConnectableProtocol)

    @skipIf(os.environ.get('CI', '').lower() == 'true' and platform.isMacOSX(), 'Flaky on macOS on Azure.')
    def test_resumeProducingAbortLater(self):
        """
        abortConnection() is called in resumeProducing, after some
        bytes have been exchanged. The protocol should be disconnected.
        """
        return self.runAbortTest(ProducerAbortingClientLater, AbortServerWritingProtocol)

    def test_fullWriteBuffer(self):
        """
        abortConnection() triggered by the write buffer being full.

        In particular, the server side stops reading. This is supposed
        to simulate a realistic timeout scenario where the client
        notices the server is no longer accepting data.

        The protocol should be disconnected, but connectionLost should not be
        called re-entrantly.
        """
        self.runAbortTest(StreamingProducerClient, NoReadServer)

    def test_fullWriteBufferAfterByteExchange(self):
        """
        abortConnection() is triggered by a write buffer being full.

        However, this buffer is filled after some bytes have been exchanged,
        allowing a TLS handshake if we're testing TLS. The connection will
        then be lost.
        """
        return self.runAbortTest(StreamingProducerClientLater, EventualNoReadServer)

    def test_dataReceivedThrows(self):
        """
        dataReceived calls abortConnection(), and then raises an exception.

        The connection will be lost, with the thrown exception
        (C{ZeroDivisionError}) as the reason on the client. The idea here is
        that bugs should not be masked by abortConnection, in particular
        unexpected exceptions.
        """
        self.runAbortTest(DataReceivedRaisingClient, AbortServerWritingProtocol, clientConnectionLostReason=ZeroDivisionError)
        errors = self.flushLoggedErrors(ZeroDivisionError)
        self.assertEqual(len(errors), 1)

    def test_resumeProducingThrows(self):
        """
        resumeProducing calls abortConnection(), and then raises an exception.

        The connection will be lost, with the thrown exception
        (C{ZeroDivisionError}) as the reason on the client. The idea here is
        that bugs should not be masked by abortConnection, in particular
        unexpected exceptions.
        """
        self.runAbortTest(ResumeThrowsClient, ConnectableProtocol, clientConnectionLostReason=ZeroDivisionError)
        errors = self.flushLoggedErrors(ZeroDivisionError)
        self.assertEqual(len(errors), 1)
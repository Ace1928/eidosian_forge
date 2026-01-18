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
class WriteSequenceTestsMixin:
    """
    Test for L{twisted.internet.abstract.FileDescriptor.writeSequence}.
    """
    requiredInterfaces: Optional[Sequence[Type[Interface]]] = (IReactorTCP,)

    def setWriteBufferSize(self, transport, value):
        """
        Set the write buffer size for the given transport, mananing possible
        differences (ie, IOCP). Bug #4322 should remove the need of that hack.
        """
        if getattr(transport, 'writeBufferSize', None) is not None:
            transport.writeBufferSize = value
        else:
            transport.bufferSize = value

    def test_writeSequeceWithoutWrite(self):
        """
        C{writeSequence} sends the data even if C{write} hasn't been called.
        """

        def connected(protocols):
            client, server, port = protocols

            def dataReceived(data):
                log.msg('data received: %r' % data)
                self.assertEqual(data, b'Some sequence splitted')
                client.transport.loseConnection()
            server.dataReceived = dataReceived
            client.transport.writeSequence([b'Some ', b'sequence ', b'splitted'])
        reactor = self.buildReactor()
        d = self.getConnectedClientAndServer(reactor, '127.0.0.1', socket.AF_INET)
        d.addCallback(connected)
        d.addErrback(log.err)
        self.runReactor(reactor)

    def test_writeSequenceWithUnicodeRaisesException(self):
        """
        C{writeSequence} with an element in the sequence of type unicode raises
        C{TypeError}.
        """

        def connected(protocols):
            client, server, port = protocols
            exc = self.assertRaises(TypeError, server.transport.writeSequence, ['Unicode is not kosher'])
            self.assertEqual(str(exc), 'Data must be bytes')
            server.transport.loseConnection()
        reactor = self.buildReactor()
        d = self.getConnectedClientAndServer(reactor, '127.0.0.1', socket.AF_INET)
        d.addCallback(connected)
        d.addErrback(log.err)
        self.runReactor(reactor)

    def test_streamingProducer(self):
        """
        C{writeSequence} pauses its streaming producer if too much data is
        buffered, and then resumes it.
        """

        @implementer(IPushProducer)
        class SaveActionProducer:
            client = None
            server = None

            def __init__(self):
                self.actions = []

            def pauseProducing(self):
                self.actions.append('pause')

            def resumeProducing(self):
                self.actions.append('resume')
                self.client.transport.unregisterProducer()
                self.server.transport.loseConnection()

            def stopProducing(self):
                self.actions.append('stop')
        producer = SaveActionProducer()

        def connected(protocols):
            client, server = protocols[:2]
            producer.client = client
            producer.server = server
            client.transport.registerProducer(producer, True)
            self.assertEqual(producer.actions, [])
            self.setWriteBufferSize(client.transport, 500)
            client.transport.writeSequence([b'x' * 50] * 20)
            self.assertEqual(producer.actions, ['pause'])
        reactor = self.buildReactor()
        d = self.getConnectedClientAndServer(reactor, '127.0.0.1', socket.AF_INET)
        d.addCallback(connected)
        d.addErrback(log.err)
        self.runReactor(reactor)
        self.assertEqual(producer.actions, ['pause', 'resume'])

    def test_nonStreamingProducer(self):
        """
        C{writeSequence} pauses its producer if too much data is buffered only
        if this is a streaming producer.
        """
        test = self

        @implementer(IPullProducer)
        class SaveActionProducer:
            client = None

            def __init__(self):
                self.actions = []

            def resumeProducing(self):
                self.actions.append('resume')
                if self.actions.count('resume') == 2:
                    self.client.transport.stopConsuming()
                else:
                    test.setWriteBufferSize(self.client.transport, 500)
                    self.client.transport.writeSequence([b'x' * 50] * 20)

            def stopProducing(self):
                self.actions.append('stop')
        producer = SaveActionProducer()

        def connected(protocols):
            client = protocols[0]
            producer.client = client
            client.transport.registerProducer(producer, False)
            self.assertEqual(producer.actions, ['resume'])
        reactor = self.buildReactor()
        d = self.getConnectedClientAndServer(reactor, '127.0.0.1', socket.AF_INET)
        d.addCallback(connected)
        d.addErrback(log.err)
        self.runReactor(reactor)
        self.assertEqual(producer.actions, ['resume', 'resume'])
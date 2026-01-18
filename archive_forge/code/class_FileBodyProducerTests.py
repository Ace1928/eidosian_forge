from __future__ import annotations
import zlib
from http.cookiejar import CookieJar
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple
from unittest import SkipTest, skipIf
from zope.interface.declarations import implementer
from zope.interface.verify import verifyObject
from incremental import Version
from twisted.internet import defer, task
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.endpoints import HostnameEndpoint, TCP4ClientEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import IOpenSSLClientConnectionCreator
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.test.test_endpoints import deterministicResolvingReactor
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import getDeprecationWarningString
from twisted.python.failure import Failure
from twisted.test.iosim import FakeTransport, IOPump
from twisted.test.test_sslverify import certificatesForAuthorityAndServer
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import client, error, http_headers
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.error import SchemeNotSupported
from twisted.web.http_headers import Headers
from twisted.web.iweb import (
from twisted.web.test.injectionhelpers import (
class FileBodyProducerTests(TestCase):
    """
    Tests for the L{FileBodyProducer} which reads bytes from a file and writes
    them to an L{IConsumer}.
    """

    def _termination(self):
        """
        This method can be used as the C{terminationPredicateFactory} for a
        L{Cooperator}.  It returns a predicate which immediately returns
        C{False}, indicating that no more work should be done this iteration.
        This has the result of only allowing one iteration of a cooperative
        task to be run per L{Cooperator} iteration.
        """
        return lambda: True

    def setUp(self):
        """
        Create a L{Cooperator} hooked up to an easily controlled, deterministic
        scheduler to use with L{FileBodyProducer}.
        """
        self._scheduled = []
        self.cooperator = task.Cooperator(self._termination, self._scheduled.append)

    def test_interface(self):
        """
        L{FileBodyProducer} instances provide L{IBodyProducer}.
        """
        self.assertTrue(verifyObject(IBodyProducer, FileBodyProducer(BytesIO(b''))))

    def test_unknownLength(self):
        """
        If the L{FileBodyProducer} is constructed with a file-like object
        without either a C{seek} or C{tell} method, its C{length} attribute is
        set to C{UNKNOWN_LENGTH}.
        """

        class HasSeek:

            def seek(self, offset, whence):
                pass

        class HasTell:

            def tell(self):
                pass
        producer = FileBodyProducer(HasSeek())
        self.assertEqual(UNKNOWN_LENGTH, producer.length)
        producer = FileBodyProducer(HasTell())
        self.assertEqual(UNKNOWN_LENGTH, producer.length)

    def test_knownLength(self):
        """
        If the L{FileBodyProducer} is constructed with a file-like object with
        both C{seek} and C{tell} methods, its C{length} attribute is set to the
        size of the file as determined by those methods.
        """
        inputBytes = b'here are some bytes'
        inputFile = BytesIO(inputBytes)
        inputFile.seek(5)
        producer = FileBodyProducer(inputFile)
        self.assertEqual(len(inputBytes) - 5, producer.length)
        self.assertEqual(inputFile.tell(), 5)

    def test_defaultCooperator(self):
        """
        If no L{Cooperator} instance is passed to L{FileBodyProducer}, the
        global cooperator is used.
        """
        producer = FileBodyProducer(BytesIO(b''))
        self.assertEqual(task.cooperate, producer._cooperate)

    def test_startProducing(self):
        """
        L{FileBodyProducer.startProducing} starts writing bytes from the input
        file to the given L{IConsumer} and returns a L{Deferred} which fires
        when they have all been written.
        """
        expectedResult = b'hello, world'
        readSize = 3
        output = BytesIO()
        consumer = FileConsumer(output)
        producer = FileBodyProducer(BytesIO(expectedResult), self.cooperator, readSize)
        complete = producer.startProducing(consumer)
        for i in range(len(expectedResult) // readSize + 1):
            self._scheduled.pop(0)()
        self.assertEqual([], self._scheduled)
        self.assertEqual(expectedResult, output.getvalue())
        self.assertEqual(None, self.successResultOf(complete))

    def test_inputClosedAtEOF(self):
        """
        When L{FileBodyProducer} reaches end-of-file on the input file given to
        it, the input file is closed.
        """
        readSize = 4
        inputBytes = b'some friendly bytes'
        inputFile = BytesIO(inputBytes)
        producer = FileBodyProducer(inputFile, self.cooperator, readSize)
        consumer = FileConsumer(BytesIO())
        producer.startProducing(consumer)
        for i in range(len(inputBytes) // readSize + 2):
            self._scheduled.pop(0)()
        self.assertTrue(inputFile.closed)

    def test_failedReadWhileProducing(self):
        """
        If a read from the input file fails while producing bytes to the
        consumer, the L{Deferred} returned by
        L{FileBodyProducer.startProducing} fires with a L{Failure} wrapping
        that exception.
        """

        class BrokenFile:

            def read(self, count):
                raise OSError('Simulated bad thing')
        producer = FileBodyProducer(BrokenFile(), self.cooperator)
        complete = producer.startProducing(FileConsumer(BytesIO()))
        self._scheduled.pop(0)()
        self.failureResultOf(complete).trap(IOError)

    def test_cancelWhileProducing(self):
        """
        When the L{Deferred} returned by L{FileBodyProducer.startProducing} is
        cancelled, the input file is closed and the task is stopped.
        """
        expectedResult = b'hello, world'
        readSize = 3
        output = BytesIO()
        consumer = FileConsumer(output)
        inputFile = BytesIO(expectedResult)
        producer = FileBodyProducer(inputFile, self.cooperator, readSize)
        complete = producer.startProducing(consumer)
        complete.cancel()
        self.assertTrue(inputFile.closed)
        self._scheduled.pop(0)()
        self.assertEqual(b'', output.getvalue())
        self.assertNoResult(complete)

    def test_stopProducing(self):
        """
        L{FileBodyProducer.stopProducing} stops the underlying L{IPullProducer}
        and the cooperative task responsible for calling C{resumeProducing} and
        closes the input file but does not cause the L{Deferred} returned by
        C{startProducing} to fire.
        """
        expectedResult = b'hello, world'
        readSize = 3
        output = BytesIO()
        consumer = FileConsumer(output)
        inputFile = BytesIO(expectedResult)
        producer = FileBodyProducer(inputFile, self.cooperator, readSize)
        complete = producer.startProducing(consumer)
        producer.stopProducing()
        self.assertTrue(inputFile.closed)
        self._scheduled.pop(0)()
        self.assertEqual(b'', output.getvalue())
        self.assertNoResult(complete)

    def test_pauseProducing(self):
        """
        L{FileBodyProducer.pauseProducing} temporarily suspends writing bytes
        from the input file to the given L{IConsumer}.
        """
        expectedResult = b'hello, world'
        readSize = 5
        output = BytesIO()
        consumer = FileConsumer(output)
        producer = FileBodyProducer(BytesIO(expectedResult), self.cooperator, readSize)
        complete = producer.startProducing(consumer)
        self._scheduled.pop(0)()
        self.assertEqual(output.getvalue(), expectedResult[:5])
        producer.pauseProducing()
        self._scheduled.pop(0)()
        self.assertEqual(output.getvalue(), expectedResult[:5])
        self.assertEqual([], self._scheduled)
        self.assertNoResult(complete)

    def test_resumeProducing(self):
        """
        L{FileBodyProducer.resumeProducing} re-commences writing bytes from the
        input file to the given L{IConsumer} after it was previously paused
        with L{FileBodyProducer.pauseProducing}.
        """
        expectedResult = b'hello, world'
        readSize = 5
        output = BytesIO()
        consumer = FileConsumer(output)
        producer = FileBodyProducer(BytesIO(expectedResult), self.cooperator, readSize)
        producer.startProducing(consumer)
        self._scheduled.pop(0)()
        self.assertEqual(expectedResult[:readSize], output.getvalue())
        producer.pauseProducing()
        producer.resumeProducing()
        self._scheduled.pop(0)()
        self.assertEqual(expectedResult[:readSize * 2], output.getvalue())

    def test_multipleStop(self):
        """
        L{FileBodyProducer.stopProducing} can be called more than once without
        raising an exception.
        """
        expectedResult = b'test'
        readSize = 3
        output = BytesIO()
        consumer = FileConsumer(output)
        inputFile = BytesIO(expectedResult)
        producer = FileBodyProducer(inputFile, self.cooperator, readSize)
        complete = producer.startProducing(consumer)
        producer.stopProducing()
        producer.stopProducing()
        self.assertTrue(inputFile.closed)
        self._scheduled.pop(0)()
        self.assertEqual(b'', output.getvalue())
        self.assertNoResult(complete)
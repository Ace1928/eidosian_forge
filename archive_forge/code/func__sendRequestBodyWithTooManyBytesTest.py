from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
def _sendRequestBodyWithTooManyBytesTest(self, finisher):
    """
        Verify that when too many bytes have been written by a body producer
        and then the body producer's C{startProducing} L{Deferred} fires that
        the producer is unregistered from the transport and that the
        L{Deferred} returned from L{Request.writeTo} is fired with a L{Failure}
        wrapping a L{WrongBodyLength}.

        @param finisher: A callable which will be invoked with the body
            producer after too many bytes have been written to the transport.
            It should fire the startProducing Deferred somehow.
        """
    producer = StringProducer(3)
    request = Request(b'POST', b'/bar', _boringHeaders, producer)
    writeDeferred = request.writeTo(self.transport)
    producer.consumer.write(b'ab')
    self.assertFalse(producer.stopped)
    producer.consumer.write(b'cd')
    self.assertTrue(producer.stopped)
    self.assertIdentical(self.transport.producer, None)

    def cbFailed(exc):
        self.assertEqual(self.transport.value(), b'POST /bar HTTP/1.1\r\nConnection: close\r\nContent-Length: 3\r\nHost: example.com\r\n\r\nab')
        self.transport.clear()
        self.assertRaises(ExcessWrite, producer.consumer.write, b'ef')
        finisher(producer)
        self.assertEqual(self.transport.value(), b'')
    d = self.assertFailure(writeDeferred, WrongBodyLength)
    d.addCallback(cbFailed)
    return d
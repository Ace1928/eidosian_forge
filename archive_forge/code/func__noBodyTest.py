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
def _noBodyTest(self, request, status, response):
    """
        Assert that L{HTTPClientParser} parses the given C{response} to
        C{request}, resulting in a response with no body and no extra bytes and
        leaving the transport in the producing state.

        @param request: A L{Request} instance which might have caused a server
            to return the given response.
        @param status: A string giving the status line of the response to be
            parsed.
        @param response: A string giving the response to be parsed.

        @return: A C{dict} of headers from the response.
        """
    header = {}
    finished = []
    body = []
    bodyDataFinished = []
    protocol = HTTPClientParser(request, finished.append)
    protocol.headerReceived = header.__setitem__
    transport = StringTransport()
    protocol.makeConnection(transport)
    protocol.dataReceived(status)
    protocol.response._bodyDataReceived = body.append
    protocol.response._bodyDataFinished = lambda: bodyDataFinished.append(True)
    protocol.dataReceived(response)
    self.assertEqual(transport.producerState, 'producing')
    self.assertEqual(protocol.state, DONE)
    self.assertEqual(body, [])
    self.assertEqual(finished, [b''])
    self.assertEqual(bodyDataFinished, [True])
    self.assertEqual(protocol.response.length, 0)
    return header
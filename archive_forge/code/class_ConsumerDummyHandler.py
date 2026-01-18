import itertools
from zope.interface import directlyProvides, providedBy
from twisted.internet import defer, error, reactor, task
from twisted.internet.address import IPv4Address
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.web import http
from twisted.web.test.test_http import (
class ConsumerDummyHandler(http.Request):
    """
    This is a HTTP request handler that works with the C{IPushProducer}
    implementation in the L{H2Stream} object. No current IRequest object does
    that, but in principle future implementations could: that codepath should
    therefore be tested.
    """

    def __init__(self, *args, **kwargs):
        http.Request.__init__(self, *args, **kwargs)
        self.channel.pauseProducing()
        self._requestReceived = False
        self._data = None

    def acceptData(self):
        """
        Start the data pipe.
        """
        self.channel.resumeProducing()

    def requestReceived(self, *args, **kwargs):
        self._requestReceived = True
        return http.Request.requestReceived(self, *args, **kwargs)

    def process(self):
        self.setResponseCode(200)
        self._data = self.content.read()
        returnData = b'this is a response from a consumer dummy handler'
        self.write(returnData)
        self.finish()
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
class ChunkedHTTPHandler(http.Request):
    """
    A HTTP request object that writes chunks of data back to the network based
    on the URL.

    Must be called with a path /chunked/<num_chunks>
    """
    chunkData = b'hello world!'

    def process(self):
        chunks = int(self.uri.split(b'/')[-1])
        self.setResponseCode(200)
        for _ in range(chunks):
            self.write(self.chunkData)
        self.finish()
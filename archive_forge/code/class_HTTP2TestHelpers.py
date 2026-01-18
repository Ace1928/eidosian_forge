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
class HTTP2TestHelpers:
    """
    A superclass that contains no tests but provides test helpers for HTTP/2
    tests.
    """
    if skipH2:
        skip = skipH2

    def assertAllStreamsBlocked(self, connection):
        """
        Confirm that all streams are blocked: that is, the priority tree
        believes that none of the streams have data ready to send.
        """
        self.assertRaises(priority.DeadlockError, next, connection.priority)
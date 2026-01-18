import datetime
from io import BytesIO, StringIO
from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver, MemoryReactor
from twisted.logger import (
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
from twisted.python.reflect import namedModule
from twisted.trial import unittest
from twisted.web import client, http, server, static, xmlrpc
from twisted.web.test.test_web import DummyRequest
from twisted.web.xmlrpc import (
class QueryFactoryParseResponseTests(unittest.TestCase):
    """
    Test the behaviour of L{QueryFactory.parseResponse}.
    """

    def setUp(self):
        self.queryFactory = QueryFactory(path=None, host=None, method='POST', user=None, password=None, allowNone=False, args=())
        self.goodContents = xmlrpclib.dumps(('',))
        self.badContents = 'invalid xml'
        self.reason = failure.Failure(ConnectionDone())

    def test_parseResponseCallbackSafety(self):
        """
        We can safely call L{QueryFactory.clientConnectionLost} as a callback
        of L{QueryFactory.parseResponse}.
        """
        d = self.queryFactory.deferred
        d.addCallback(self.queryFactory.clientConnectionLost, self.reason)
        self.queryFactory.parseResponse(self.goodContents)
        return d

    def test_parseResponseErrbackSafety(self):
        """
        We can safely call L{QueryFactory.clientConnectionLost} as an errback
        of L{QueryFactory.parseResponse}.
        """
        d = self.queryFactory.deferred
        d.addErrback(self.queryFactory.clientConnectionLost, self.reason)
        self.queryFactory.parseResponse(self.badContents)
        return d

    def test_badStatusErrbackSafety(self):
        """
        We can safely call L{QueryFactory.clientConnectionLost} as an errback
        of L{QueryFactory.badStatus}.
        """
        d = self.queryFactory.deferred
        d.addErrback(self.queryFactory.clientConnectionLost, self.reason)
        self.queryFactory.badStatus('status', 'message')
        return d

    def test_parseResponseWithoutData(self):
        """
        Some server can send a response without any data:
        L{QueryFactory.parseResponse} should catch the error and call the
        result errback.
        """
        content = '\n<methodResponse>\n <params>\n  <param>\n  </param>\n </params>\n</methodResponse>'
        d = self.queryFactory.deferred
        self.queryFactory.parseResponse(content)
        return self.assertFailure(d, IndexError)
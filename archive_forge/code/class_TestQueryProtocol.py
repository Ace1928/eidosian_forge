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
class TestQueryProtocol(xmlrpc.QueryProtocol):
    """
    QueryProtocol for tests that saves headers received and sent,
    inside the factory.
    """

    def connectionMade(self):
        self.factory.transport = self.transport
        xmlrpc.QueryProtocol.connectionMade(self)

    def handleHeader(self, key, val):
        self.factory.headers[key.lower()] = val

    def sendHeader(self, key, val):
        """
        Keep sent headers so we can inspect them later.
        """
        self.factory.sent_headers[key.lower()] = val
        xmlrpc.QueryProtocol.sendHeader(self, key, val)
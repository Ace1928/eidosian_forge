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
class XMLRPCProxyWithoutSlashTests(XMLRPCTests):
    """
    Test with proxy that doesn't add a slash.
    """

    def proxy(self, factory=None):
        p = xmlrpc.Proxy(networkString('http://127.0.0.1:%d' % self.port))
        if factory is None:
            p.queryFactory = self.queryFactory
        else:
            p.queryFactory = factory
        return p
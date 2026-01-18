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
class SerializationConfigMixin:
    """
    Mixin which defines a couple tests which should pass when a particular flag
    is passed to L{XMLRPC}.

    These are not meant to be exhaustive serialization tests, since L{xmlrpclib}
    does all of the actual serialization work.  They are just meant to exercise
    a few codepaths to make sure we are calling into xmlrpclib correctly.

    @ivar flagName: A C{str} giving the name of the flag which must be passed to
        L{XMLRPC} to allow the tests to pass.  Subclasses should set this.

    @ivar value: A value which the specified flag will allow the serialization
        of.  Subclasses should set this.
    """

    def setUp(self):
        """
        Create a new XML-RPC server with C{allowNone} set to C{True}.
        """
        kwargs = {self.flagName: True}
        self.p = reactor.listenTCP(0, server.Site(Test(**kwargs)), interface='127.0.0.1')
        self.addCleanup(self.p.stopListening)
        self.port = self.p.getHost().port
        self.proxy = xmlrpc.Proxy(networkString('http://127.0.0.1:%d/' % (self.port,)), **kwargs)

    def test_roundtripValue(self):
        """
        C{self.value} can be round-tripped over an XMLRPC method call/response.
        """
        d = self.proxy.callRemote('defer', self.value)
        d.addCallback(self.assertEqual, self.value)
        return d

    def test_roundtripNestedValue(self):
        """
        A C{dict} which contains C{self.value} can be round-tripped over an
        XMLRPC method call/response.
        """
        d = self.proxy.callRemote('defer', {'a': self.value})
        d.addCallback(self.assertEqual, {'a': self.value})
        return d
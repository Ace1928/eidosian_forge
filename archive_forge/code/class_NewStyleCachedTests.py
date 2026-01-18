import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest
class NewStyleCachedTests(unittest.TestCase):

    def setUp(self):
        """
        Create a pb server using L{CachedReturner} protocol and connect a
        client to it.
        """
        self.orig = NewStyleCacheCopy()
        self.orig.s = 'value'
        self.server = reactor.listenTCP(0, ConnectionNotifyServerFactory(CachedReturner(self.orig)))
        clientFactory = pb.PBClientFactory()
        reactor.connectTCP('localhost', self.server.getHost().port, clientFactory)

        def gotRoot(ref):
            self.ref = ref
        d1 = clientFactory.getRootObject().addCallback(gotRoot)
        d2 = self.server.factory.connectionMade
        return gatherResults([d1, d2])

    def tearDown(self):
        """
        Close client and server connections.
        """
        self.server.factory.protocolInstance.transport.loseConnection()
        self.ref.broker.transport.loseConnection()
        return self.server.stopListening()

    def test_newStyleCache(self):
        """
        A new-style cacheable object can be retrieved and re-retrieved over a
        single connection.  The value of an attribute of the cacheable can be
        accessed on the receiving side.
        """
        d = self.ref.callRemote('giveMeCache', self.orig)

        def cb(res, again):
            self.assertIsInstance(res, NewStyleCacheCopy)
            self.assertEqual('value', res.s)
            self.assertIsNot(self.orig, res)
            if again:
                self.res = res
                return self.ref.callRemote('giveMeCache', self.orig)
        d.addCallback(cb, True)
        d.addCallback(cb, False)
        return d
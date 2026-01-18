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
class NSPTests(unittest.TestCase):
    """
    Tests for authentication against a realm where the L{IPerspective}
    implementation is not a subclass of L{Avatar}.
    """

    def setUp(self):
        self.realm = TestRealm()
        self.realm.perspectiveFactory = NonSubclassingPerspective
        self.portal = portal.Portal(self.realm)
        self.checker = checkers.InMemoryUsernamePasswordDatabaseDontUse()
        self.checker.addUser(b'user', b'pass')
        self.portal.registerChecker(self.checker)
        self.factory = WrappingFactory(pb.PBServerFactory(self.portal))
        self.port = reactor.listenTCP(0, self.factory, interface='127.0.0.1')
        self.addCleanup(self.port.stopListening)
        self.portno = self.port.getHost().port

    def test_NSP(self):
        """
        An L{IPerspective} implementation which does not subclass
        L{Avatar} can expose remote methods for the client to call.
        """
        factory = pb.PBClientFactory()
        d = factory.login(credentials.UsernamePassword(b'user', b'pass'), 'BRAINS!')
        reactor.connectTCP('127.0.0.1', self.portno, factory)
        d.addCallback(lambda p: p.callRemote('ANYTHING', 'here', bar='baz'))
        d.addCallback(self.assertEqual, ('ANYTHING', ('here',), {'bar': 'baz'}))

        def cleanup(ignored):
            factory.disconnect()
            for p in self.factory.protocols:
                p.transport.loseConnection()
        d.addCallback(cleanup)
        return d
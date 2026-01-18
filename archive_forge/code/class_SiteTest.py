import os
import zlib
from io import BytesIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import interfaces
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.task import Clock
from twisted.internet.testing import EventLoggingObserver, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python import failure, reflect
from twisted.python.compat import iterbytes
from twisted.python.filepath import FilePath
from twisted.trial import unittest
from twisted.web import error, http, iweb, resource, server
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Request, Site
from twisted.web.static import Data
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from ._util import assertIsFilesystemTemporary
class SiteTest(unittest.TestCase):
    """
    Unit tests for L{server.Site}.
    """

    def getAutoExpiringSession(self, site):
        """
        Create a new session which auto expires at cleanup.

        @param site: The site on which the session is created.
        @type site: L{server.Site}

        @return: A newly created session.
        @rtype: L{server.Session}
        """
        session = site.makeSession()
        self.addCleanup(session.expire)
        return session

    def test_simplestSite(self):
        """
        L{Site.getResourceFor} returns the C{b""} child of the root resource it
        is constructed with when processing a request for I{/}.
        """
        sres1 = SimpleResource()
        sres2 = SimpleResource()
        sres1.putChild(b'', sres2)
        site = server.Site(sres1)
        self.assertIdentical(site.getResourceFor(DummyRequest([b''])), sres2, 'Got the wrong resource.')

    def test_defaultRequestFactory(self):
        """
        L{server.Request} is the default request factory.
        """
        site = server.Site(resource=SimpleResource())
        self.assertIs(server.Request, site.requestFactory)

    def test_constructorRequestFactory(self):
        """
        Can be initialized with a custom requestFactory.
        """
        customFactory = object()
        site = server.Site(resource=SimpleResource(), requestFactory=customFactory)
        self.assertIs(customFactory, site.requestFactory)

    def test_buildProtocol(self):
        """
        Returns a C{Channel} whose C{site} and C{requestFactory} attributes are
        assigned from the C{site} instance.
        """
        site = server.Site(SimpleResource())
        channel = site.buildProtocol(None)
        self.assertIs(site, channel.site)
        self.assertIs(site.requestFactory, channel.requestFactory)

    def test_makeSession(self):
        """
        L{site.getSession} generates a new C{Session} instance with an uid of
        type L{bytes}.
        """
        site = server.Site(resource.Resource())
        session = self.getAutoExpiringSession(site)
        self.assertIsInstance(session, server.Session)
        self.assertIsInstance(session.uid, bytes)

    def test_sessionUIDGeneration(self):
        """
        L{site.getSession} generates L{Session} objects with distinct UIDs from
        a secure source of entropy.
        """
        site = server.Site(resource.Resource())
        self.assertIdentical(site._entropy, os.urandom)

        def predictableEntropy(n):
            predictableEntropy.x += 1
            return (chr(predictableEntropy.x) * n).encode('charmap')
        predictableEntropy.x = 0
        self.patch(site, '_entropy', predictableEntropy)
        a = self.getAutoExpiringSession(site)
        b = self.getAutoExpiringSession(site)
        self.assertEqual(a.uid, b'01' * 32)
        self.assertEqual(b.uid, b'02' * 32)
        self.assertEqual(site.counter, 2)

    def test_getSessionExistent(self):
        """
        L{site.getSession} gets a previously generated session, by its unique
        ID.
        """
        site = server.Site(resource.Resource())
        createdSession = self.getAutoExpiringSession(site)
        retrievedSession = site.getSession(createdSession.uid)
        self.assertIs(createdSession, retrievedSession)

    def test_getSessionNonExistent(self):
        """
        L{site.getSession} raises a L{KeyError} if the session is not found.
        """
        site = server.Site(resource.Resource())
        self.assertRaises(KeyError, site.getSession, b'no-such-uid')
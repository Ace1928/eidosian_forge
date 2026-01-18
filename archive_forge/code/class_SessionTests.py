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
class SessionTests(unittest.TestCase):
    """
    Tests for L{server.Session}.
    """

    def setUp(self):
        """
        Create a site with one active session using a deterministic, easily
        controlled clock.
        """
        self.clock = Clock()
        self.uid = b'unique'
        self.site = server.Site(resource.Resource(), reactor=self.clock)
        self.session = server.Session(self.site, self.uid)
        self.site.sessions[self.uid] = self.session

    def test_defaultReactor(self):
        """
        If no value is passed to L{server.Session.__init__}, the reactor
        associated with the site is used.
        """
        site = server.Site(resource.Resource(), reactor=Clock())
        session = server.Session(site, b'123')
        self.assertIdentical(session._reactor, site.reactor)

    def test_explicitReactor(self):
        """
        L{Session} accepts the reactor to use as a parameter.
        """
        site = server.Site(resource.Resource())
        otherReactor = Clock()
        session = server.Session(site, b'123', reactor=otherReactor)
        self.assertIdentical(session._reactor, otherReactor)

    def test_startCheckingExpiration(self):
        """
        L{server.Session.startCheckingExpiration} causes the session to expire
        after L{server.Session.sessionTimeout} seconds without activity.
        """
        self.session.startCheckingExpiration()
        self.clock.advance(self.session.sessionTimeout - 1)
        self.assertIn(self.uid, self.site.sessions)
        self.clock.advance(1)
        self.assertNotIn(self.uid, self.site.sessions)
        self.assertFalse(self.clock.calls)

    def test_expire(self):
        """
        L{server.Session.expire} expires the session.
        """
        self.session.expire()
        self.assertNotIn(self.uid, self.site.sessions)
        self.assertFalse(self.clock.calls)

    def test_expireWhileChecking(self):
        """
        L{server.Session.expire} expires the session even if the timeout call
        isn't due yet.
        """
        self.session.startCheckingExpiration()
        self.test_expire()

    def test_notifyOnExpire(self):
        """
        A function registered with L{server.Session.notifyOnExpire} is called
        when the session expires.
        """
        callbackRan = [False]

        def expired():
            callbackRan[0] = True
        self.session.notifyOnExpire(expired)
        self.session.expire()
        self.assertTrue(callbackRan[0])

    def test_touch(self):
        """
        L{server.Session.touch} updates L{server.Session.lastModified} and
        delays session timeout.
        """
        self.clock.advance(3)
        self.session.touch()
        self.assertEqual(self.session.lastModified, 3)
        self.session.startCheckingExpiration()
        self.clock.advance(self.session.sessionTimeout - 1)
        self.session.touch()
        self.clock.advance(self.session.sessionTimeout - 1)
        self.assertIn(self.uid, self.site.sessions)
        self.clock.advance(1)
        self.assertNotIn(self.uid, self.site.sessions)
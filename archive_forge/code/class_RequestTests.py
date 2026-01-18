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
class RequestTests(unittest.TestCase):
    """
    Tests for the HTTP request class, L{server.Request}.
    """

    def test_interface(self):
        """
        L{server.Request} instances provide L{iweb.IRequest}.
        """
        self.assertTrue(verifyObject(iweb.IRequest, server.Request(DummyChannel(), True)))

    def test_hashable(self):
        """
        L{server.Request} instances are hashable, thus can be put in a mapping.
        """
        request = server.Request(DummyChannel(), True)
        hash(request)

    def testChildLink(self):
        request = server.Request(DummyChannel(), 1)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.childLink(b'baz'), b'bar/baz')
        request = server.Request(DummyChannel(), 1)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar/', b'HTTP/1.0')
        self.assertEqual(request.childLink(b'baz'), b'baz')

    def testPrePathURLSimple(self):
        request = server.Request(DummyChannel(), 1)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        request.setHost(b'example.com', 80)
        self.assertEqual(request.prePathURL(), b'http://example.com/foo/bar')

    def testPrePathURLNonDefault(self):
        d = DummyChannel()
        d.transport.port = 81
        request = server.Request(d, 1)
        request.setHost(b'example.com', 81)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'http://example.com:81/foo/bar')

    def testPrePathURLSSLPort(self):
        d = DummyChannel()
        d.transport.port = 443
        request = server.Request(d, 1)
        request.setHost(b'example.com', 443)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'http://example.com:443/foo/bar')

    def testPrePathURLSSLPortAndSSL(self):
        d = DummyChannel()
        d.transport = DummyChannel.SSL()
        d.transport.port = 443
        request = server.Request(d, 1)
        request.setHost(b'example.com', 443)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'https://example.com/foo/bar')

    def testPrePathURLHTTPPortAndSSL(self):
        d = DummyChannel()
        d.transport = DummyChannel.SSL()
        d.transport.port = 80
        request = server.Request(d, 1)
        request.setHost(b'example.com', 80)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'https://example.com:80/foo/bar')

    def testPrePathURLSSLNonDefault(self):
        d = DummyChannel()
        d.transport = DummyChannel.SSL()
        d.transport.port = 81
        request = server.Request(d, 1)
        request.setHost(b'example.com', 81)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'https://example.com:81/foo/bar')

    def testPrePathURLSetSSLHost(self):
        d = DummyChannel()
        d.transport.port = 81
        request = server.Request(d, 1)
        request.setHost(b'foo.com', 81, 1)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'https://foo.com:81/foo/bar')

    def test_prePathURLQuoting(self):
        """
        L{Request.prePathURL} quotes special characters in the URL segments to
        preserve the original meaning.
        """
        d = DummyChannel()
        request = server.Request(d, 1)
        request.setHost(b'example.com', 80)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo%2Fbar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'http://example.com/foo%2Fbar')

    def test_processingFailedNoTracebackByDefault(self):
        """
        By default, L{Request.processingFailed} does not write out the failure,
        but give a generic error message, as L{Site.displayTracebacks} is
        disabled by default.
        """
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = server.Site(resource.Resource())
        fail = failure.Failure(Exception('Oh no!'))
        request.processingFailed(fail)
        self.assertNotIn(b'Oh no!', request.transport.written.getvalue())
        self.assertIn(b'Processing Failed', request.transport.written.getvalue())
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        f = event['log_failure']
        self.assertIsInstance(f.value, Exception)
        self.assertEquals(f.getErrorMessage(), 'Oh no!')
        self.assertEqual(1, len(self.flushLoggedErrors()))

    def test_processingFailedNoTraceback(self):
        """
        L{Request.processingFailed} when the site has C{displayTracebacks} set
        to C{False} does not write out the failure, but give a generic error
        message.
        """
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = server.Site(resource.Resource())
        request.site.displayTracebacks = False
        fail = failure.Failure(Exception('Oh no!'))
        request.processingFailed(fail)
        self.assertNotIn(b'Oh no!', request.transport.written.getvalue())
        self.assertIn(b'Processing Failed', request.transport.written.getvalue())
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        f = event['log_failure']
        self.assertIsInstance(f.value, Exception)
        self.assertEquals(f.getErrorMessage(), 'Oh no!')
        self.assertEqual(1, len(self.flushLoggedErrors()))

    def test_processingFailedDisplayTraceback(self):
        """
        L{Request.processingFailed} when the site has C{displayTracebacks} set
        to C{True} writes out the failure.
        """
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = server.Site(resource.Resource())
        request.site.displayTracebacks = True
        fail = failure.Failure(Exception('Oh no!'))
        request.processingFailed(fail)
        self.assertIn(b'Oh no!', request.transport.written.getvalue())
        event = logObserver[0]
        f = event['log_failure']
        self.assertIsInstance(f.value, Exception)
        self.assertEquals(f.getErrorMessage(), 'Oh no!')
        self.assertEqual(1, len(self.flushLoggedErrors()))

    def test_processingFailedDisplayTracebackHandlesUnicode(self):
        """
        L{Request.processingFailed} when the site has C{displayTracebacks} set
        to C{True} writes out the failure, making UTF-8 items into HTML
        entities.
        """
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = server.Site(resource.Resource())
        request.site.displayTracebacks = True
        fail = failure.Failure(Exception('☃'))
        request.processingFailed(fail)
        self.assertIn(b'&#9731;', request.transport.written.getvalue())
        self.flushLoggedErrors(UnicodeError)
        event = logObserver[0]
        f = event['log_failure']
        self.assertIsInstance(f.value, Exception)
        self.assertEqual(1, len(self.flushLoggedErrors()))

    def test_sessionDifferentFromSecureSession(self):
        """
        L{Request.session} and L{Request.secure_session} should be two separate
        sessions with unique ids and different cookies.
        """
        d = DummyChannel()
        d.transport = DummyChannel.SSL()
        request = server.Request(d, 1)
        request.site = server.Site(resource.Resource())
        request.sitepath = []
        secureSession = request.getSession()
        self.assertIsNotNone(secureSession)
        self.addCleanup(secureSession.expire)
        self.assertEqual(request.cookies[0].split(b'=')[0], b'TWISTED_SECURE_SESSION')
        session = request.getSession(forceNotSecure=True)
        self.assertIsNotNone(session)
        self.assertEqual(request.cookies[1].split(b'=')[0], b'TWISTED_SESSION')
        self.addCleanup(session.expire)
        self.assertNotEqual(session.uid, secureSession.uid)

    def test_sessionAttribute(self):
        """
        On a L{Request}, the C{session} attribute retrieves the associated
        L{Session} only if it has been initialized.  If the request is secure,
        it retrieves the secure session.
        """
        site = server.Site(resource.Resource())
        d = DummyChannel()
        d.transport = DummyChannel.SSL()
        request = server.Request(d, 1)
        request.site = site
        request.sitepath = []
        self.assertIs(request.session, None)
        insecureSession = request.getSession(forceNotSecure=True)
        self.addCleanup(insecureSession.expire)
        self.assertIs(request.session, None)
        secureSession = request.getSession()
        self.addCleanup(secureSession.expire)
        self.assertIsNot(secureSession, None)
        self.assertIsNot(secureSession, insecureSession)
        self.assertIs(request.session, secureSession)

    def test_sessionCaching(self):
        """
        L{Request.getSession} creates the session object only once per request;
        if it is called twice it returns the identical result.
        """
        site = server.Site(resource.Resource())
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = site
        request.sitepath = []
        session1 = request.getSession()
        self.addCleanup(session1.expire)
        session2 = request.getSession()
        self.assertIs(session1, session2)

    def test_retrieveExistingSession(self):
        """
        L{Request.getSession} retrieves an existing session if the relevant
        cookie is set in the incoming request.
        """
        site = server.Site(resource.Resource())
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = site
        request.sitepath = []
        mySession = server.Session(site, b'special-id')
        site.sessions[mySession.uid] = mySession
        request.received_cookies[b'TWISTED_SESSION'] = mySession.uid
        self.assertIs(request.getSession(), mySession)

    def test_retrieveNonExistentSession(self):
        """
        L{Request.getSession} generates a new session if the session ID
        advertised in the cookie from the incoming request is not found.
        """
        site = server.Site(resource.Resource())
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = site
        request.sitepath = []
        request.received_cookies[b'TWISTED_SESSION'] = b'does-not-exist'
        session = request.getSession()
        self.assertIsNotNone(session)
        self.addCleanup(session.expire)
        self.assertTrue(request.cookies[0].startswith(b'TWISTED_SESSION='))
        self.assertNotIn(b'does-not-exist', request.cookies[0])

    def test_getSessionExpired(self):
        """
        L{Request.getSession} generates a new session when the previous
        session has expired.
        """
        clock = Clock()
        site = server.Site(resource.Resource())
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = site
        request.sitepath = []

        def sessionFactoryWithClock(site, uid):
            """
            Forward to normal session factory, but inject the clock.

            @param site: The site on which the session is created.
            @type site: L{server.Site}

            @param uid: A unique identifier for the session.
            @type uid: C{bytes}

            @return: A newly created session.
            @rtype: L{server.Session}
            """
            session = sessionFactory(site, uid)
            session._reactor = clock
            return session
        sessionFactory = site.sessionFactory
        site.sessionFactory = sessionFactoryWithClock
        initialSession = request.getSession()
        clock.advance(sessionFactory.sessionTimeout)
        newSession = request.getSession()
        self.addCleanup(newSession.expire)
        self.assertIsNot(initialSession, newSession)
        self.assertNotEqual(initialSession.uid, newSession.uid)

    def test_OPTIONSStar(self):
        """
        L{Request} handles OPTIONS * requests by doing a fast-path return of
        200 OK.
        """
        d = DummyChannel()
        request = server.Request(d, 1)
        request.setHost(b'example.com', 80)
        request.gotLength(0)
        request.requestReceived(b'OPTIONS', b'*', b'HTTP/1.1')
        response = d.transport.written.getvalue()
        self.assertTrue(response.startswith(b'HTTP/1.1 200 OK'))
        self.assertIn(b'Content-Length: 0\r\n', response)

    def test_rejectNonOPTIONSStar(self):
        """
        L{Request} handles any non-OPTIONS verb requesting the * path by doing
        a fast-return 405 Method Not Allowed, indicating only the support for
        OPTIONS.
        """
        d = DummyChannel()
        request = server.Request(d, 1)
        request.setHost(b'example.com', 80)
        request.gotLength(0)
        request.requestReceived(b'GET', b'*', b'HTTP/1.1')
        response = d.transport.written.getvalue()
        self.assertTrue(response.startswith(b'HTTP/1.1 405 Method Not Allowed'))
        self.assertIn(b'Content-Length: 0\r\n', response)
        self.assertIn(b'Allow: OPTIONS\r\n', response)

    def test_noDefaultContentTypeOnZeroLengthResponse(self):
        """
        Responses with no length do not have a default content-type applied.
        """
        resrc = ZeroLengthResource()
        resrc.putChild(b'', resrc)
        site = server.Site(resrc)
        d = DummyChannel()
        d.site = site
        request = server.Request(d, 1)
        request.site = site
        request.setHost(b'example.com', 80)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/', b'HTTP/1.1')
        self.assertNotIn(b'content-type', request.transport.written.getvalue().lower())

    def test_noDefaultContentTypeOn204Response(self):
        """
        Responses with a 204 status code have no default content-type applied.
        """
        resrc = NoContentResource()
        resrc.putChild(b'', resrc)
        site = server.Site(resrc)
        d = DummyChannel()
        d.site = site
        request = server.Request(d, 1)
        request.site = site
        request.setHost(b'example.com', 80)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/', b'HTTP/1.1')
        response = request.transport.written.getvalue()
        self.assertTrue(response.startswith(b'HTTP/1.1 204 No Content\r\n'))
        self.assertNotIn(b'content-type', response.lower())

    def test_defaultSmallContentFile(self):
        """
        L{http.Request} creates a L{BytesIO} if the content length is small and
        the site doesn't offer to create one.
        """
        request = server.Request(DummyChannel())
        request.gotLength(100000 - 1)
        self.assertIsInstance(request.content, BytesIO)

    def test_defaultLargerContentFile(self):
        """
        L{http.Request} creates a temporary file on the filesystem if the
        content length is larger and the site doesn't offer to create one.
        """
        request = server.Request(DummyChannel())
        request.gotLength(100000)
        assertIsFilesystemTemporary(self, request.content)

    def test_defaultUnknownSizeContentFile(self):
        """
        L{http.Request} creates a temporary file on the filesystem if the
        content length is not known and the site doesn't offer to create one.
        """
        request = server.Request(DummyChannel())
        request.gotLength(None)
        assertIsFilesystemTemporary(self, request.content)

    def test_siteSuppliedContentFile(self):
        """
        L{http.Request} uses L{Site.getContentFile}, if it exists, to get a
        file-like object for the request content.
        """
        lengths = []
        contentFile = BytesIO()
        site = server.Site(resource.Resource())

        def getContentFile(length):
            lengths.append(length)
            return contentFile
        site.getContentFile = getContentFile
        channel = DummyChannel()
        channel.site = site
        request = server.Request(channel)
        request.gotLength(12345)
        self.assertEqual([12345], lengths)
        self.assertIs(contentFile, request.content)
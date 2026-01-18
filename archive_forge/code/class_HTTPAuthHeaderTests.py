import base64
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.cred import error, portal
from twisted.cred.checkers import (
from twisted.cred.credentials import IUsernamePassword
from twisted.internet.address import IPv4Address
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial import unittest
from twisted.web._auth import basic, digest
from twisted.web._auth.basic import BasicCredentialFactory
from twisted.web._auth.wrapper import HTTPAuthSessionWrapper, UnauthorizedResource
from twisted.web.iweb import ICredentialFactory
from twisted.web.resource import IResource, Resource, getChildForRequest
from twisted.web.server import NOT_DONE_YET
from twisted.web.static import Data
from twisted.web.test.test_web import DummyRequest
class HTTPAuthHeaderTests(unittest.TestCase):
    """
    Tests for L{HTTPAuthSessionWrapper}.
    """
    makeRequest = DummyRequest

    def setUp(self):
        """
        Create a realm, portal, and L{HTTPAuthSessionWrapper} to use in the tests.
        """
        self.username = b'foo bar'
        self.password = b'bar baz'
        self.avatarContent = b'contents of the avatar resource itself'
        self.childName = b'foo-child'
        self.childContent = b'contents of the foo child of the avatar'
        self.checker = InMemoryUsernamePasswordDatabaseDontUse()
        self.checker.addUser(self.username, self.password)
        self.avatar = Data(self.avatarContent, 'text/plain')
        self.avatar.putChild(self.childName, Data(self.childContent, 'text/plain'))
        self.avatars = {self.username: self.avatar}
        self.realm = Realm(self.avatars.get)
        self.portal = portal.Portal(self.realm, [self.checker])
        self.credentialFactories = []
        self.wrapper = HTTPAuthSessionWrapper(self.portal, self.credentialFactories)

    def _authorizedBasicLogin(self, request):
        """
        Add an I{basic authorization} header to the given request and then
        dispatch it, starting from C{self.wrapper} and returning the resulting
        L{IResource}.
        """
        authorization = b64encode(self.username + b':' + self.password)
        request.requestHeaders.addRawHeader(b'authorization', b'Basic ' + authorization)
        return getChildForRequest(self.wrapper, request)

    def test_getChildWithDefault(self):
        """
        Resource traversal which encounters an L{HTTPAuthSessionWrapper}
        results in an L{UnauthorizedResource} instance when the request does
        not have the required I{Authorization} headers.
        """
        request = self.makeRequest([self.childName])
        child = getChildForRequest(self.wrapper, request)
        d = request.notifyFinish()

        def cbFinished(result):
            self.assertEqual(request.responseCode, 401)
        d.addCallback(cbFinished)
        request.render(child)
        return d

    def _invalidAuthorizationTest(self, response):
        """
        Create a request with the given value as the value of an
        I{Authorization} header and perform resource traversal with it,
        starting at C{self.wrapper}.  Assert that the result is a 401 response
        code.  Return a L{Deferred} which fires when this is all done.
        """
        self.credentialFactories.append(BasicCredentialFactory('example.com'))
        request = self.makeRequest([self.childName])
        request.requestHeaders.addRawHeader(b'authorization', response)
        child = getChildForRequest(self.wrapper, request)
        d = request.notifyFinish()

        def cbFinished(result):
            self.assertEqual(request.responseCode, 401)
        d.addCallback(cbFinished)
        request.render(child)
        return d

    def test_getChildWithDefaultUnauthorizedUser(self):
        """
        Resource traversal which enouncters an L{HTTPAuthSessionWrapper}
        results in an L{UnauthorizedResource} when the request has an
        I{Authorization} header with a user which does not exist.
        """
        return self._invalidAuthorizationTest(b'Basic ' + b64encode(b'foo:bar'))

    def test_getChildWithDefaultUnauthorizedPassword(self):
        """
        Resource traversal which enouncters an L{HTTPAuthSessionWrapper}
        results in an L{UnauthorizedResource} when the request has an
        I{Authorization} header with a user which exists and the wrong
        password.
        """
        return self._invalidAuthorizationTest(b'Basic ' + b64encode(self.username + b':bar'))

    def test_getChildWithDefaultUnrecognizedScheme(self):
        """
        Resource traversal which enouncters an L{HTTPAuthSessionWrapper}
        results in an L{UnauthorizedResource} when the request has an
        I{Authorization} header with an unrecognized scheme.
        """
        return self._invalidAuthorizationTest(b'Quux foo bar baz')

    def test_getChildWithDefaultAuthorized(self):
        """
        Resource traversal which encounters an L{HTTPAuthSessionWrapper}
        results in an L{IResource} which renders the L{IResource} avatar
        retrieved from the portal when the request has a valid I{Authorization}
        header.
        """
        self.credentialFactories.append(BasicCredentialFactory('example.com'))
        request = self.makeRequest([self.childName])
        child = self._authorizedBasicLogin(request)
        d = request.notifyFinish()

        def cbFinished(ignored):
            self.assertEqual(request.written, [self.childContent])
        d.addCallback(cbFinished)
        request.render(child)
        return d

    def test_renderAuthorized(self):
        """
        Resource traversal which terminates at an L{HTTPAuthSessionWrapper}
        and includes correct authentication headers results in the
        L{IResource} avatar (not one of its children) retrieved from the
        portal being rendered.
        """
        self.credentialFactories.append(BasicCredentialFactory('example.com'))
        request = self.makeRequest([])
        child = self._authorizedBasicLogin(request)
        d = request.notifyFinish()

        def cbFinished(ignored):
            self.assertEqual(request.written, [self.avatarContent])
        d.addCallback(cbFinished)
        request.render(child)
        return d

    def test_getChallengeCalledWithRequest(self):
        """
        When L{HTTPAuthSessionWrapper} finds an L{ICredentialFactory} to issue
        a challenge, it calls the C{getChallenge} method with the request as an
        argument.
        """

        @implementer(ICredentialFactory)
        class DumbCredentialFactory:
            scheme = b'dumb'

            def __init__(self):
                self.requests = []

            def getChallenge(self, request):
                self.requests.append(request)
                return {}
        factory = DumbCredentialFactory()
        self.credentialFactories.append(factory)
        request = self.makeRequest([self.childName])
        child = getChildForRequest(self.wrapper, request)
        d = request.notifyFinish()

        def cbFinished(ignored):
            self.assertEqual(factory.requests, [request])
        d.addCallback(cbFinished)
        request.render(child)
        return d

    def _logoutTest(self):
        """
        Issue a request for an authentication-protected resource using valid
        credentials and then return the C{DummyRequest} instance which was
        used.

        This is a helper for tests about the behavior of the logout
        callback.
        """
        self.credentialFactories.append(BasicCredentialFactory('example.com'))

        class SlowerResource(Resource):

            def render(self, request):
                return NOT_DONE_YET
        self.avatar.putChild(self.childName, SlowerResource())
        request = self.makeRequest([self.childName])
        child = self._authorizedBasicLogin(request)
        request.render(child)
        self.assertEqual(self.realm.loggedOut, 0)
        return request

    def test_logout(self):
        """
        The realm's logout callback is invoked after the resource is rendered.
        """
        request = self._logoutTest()
        request.finish()
        self.assertEqual(self.realm.loggedOut, 1)

    def test_logoutOnError(self):
        """
        The realm's logout callback is also invoked if there is an error
        generating the response (for example, if the client disconnects
        early).
        """
        request = self._logoutTest()
        request.processingFailed(Failure(ConnectionDone('Simulated disconnect')))
        self.assertEqual(self.realm.loggedOut, 1)

    def test_decodeRaises(self):
        """
        Resource traversal which enouncters an L{HTTPAuthSessionWrapper}
        results in an L{UnauthorizedResource} when the request has a I{Basic
        Authorization} header which cannot be decoded using base64.
        """
        self.credentialFactories.append(BasicCredentialFactory('example.com'))
        request = self.makeRequest([self.childName])
        request.requestHeaders.addRawHeader(b'authorization', b'Basic decode should fail')
        child = getChildForRequest(self.wrapper, request)
        self.assertIsInstance(child, UnauthorizedResource)

    def test_selectParseResponse(self):
        """
        L{HTTPAuthSessionWrapper._selectParseHeader} returns a two-tuple giving
        the L{ICredentialFactory} to use to parse the header and a string
        containing the portion of the header which remains to be parsed.
        """
        basicAuthorization = b'Basic abcdef123456'
        self.assertEqual(self.wrapper._selectParseHeader(basicAuthorization), (None, None))
        factory = BasicCredentialFactory('example.com')
        self.credentialFactories.append(factory)
        self.assertEqual(self.wrapper._selectParseHeader(basicAuthorization), (factory, b'abcdef123456'))

    def test_unexpectedDecodeError(self):
        """
        Any unexpected exception raised by the credential factory's C{decode}
        method results in a 500 response code and causes the exception to be
        logged.
        """
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)

        class UnexpectedException(Exception):
            pass

        class BadFactory:
            scheme = b'bad'

            def getChallenge(self, client):
                return {}

            def decode(self, response, request):
                raise UnexpectedException()
        self.credentialFactories.append(BadFactory())
        request = self.makeRequest([self.childName])
        request.requestHeaders.addRawHeader(b'authorization', b'Bad abc')
        child = getChildForRequest(self.wrapper, request)
        request.render(child)
        self.assertEqual(request.responseCode, 500)
        self.assertEquals(1, len(logObserver))
        self.assertIsInstance(logObserver[0]['log_failure'].value, UnexpectedException)
        self.assertEqual(len(self.flushLoggedErrors(UnexpectedException)), 1)

    def test_unexpectedLoginError(self):
        """
        Any unexpected failure from L{Portal.login} results in a 500 response
        code and causes the failure to be logged.
        """
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)

        class UnexpectedException(Exception):
            pass

        class BrokenChecker:
            credentialInterfaces = (IUsernamePassword,)

            def requestAvatarId(self, credentials):
                raise UnexpectedException()
        self.portal.registerChecker(BrokenChecker())
        self.credentialFactories.append(BasicCredentialFactory('example.com'))
        request = self.makeRequest([self.childName])
        child = self._authorizedBasicLogin(request)
        request.render(child)
        self.assertEqual(request.responseCode, 500)
        self.assertEquals(1, len(logObserver))
        self.assertIsInstance(logObserver[0]['log_failure'].value, UnexpectedException)
        self.assertEqual(len(self.flushLoggedErrors(UnexpectedException)), 1)

    def test_anonymousAccess(self):
        """
        Anonymous requests are allowed if a L{Portal} has an anonymous checker
        registered.
        """
        unprotectedContents = b'contents of the unprotected child resource'
        self.avatars[ANONYMOUS] = Resource()
        self.avatars[ANONYMOUS].putChild(self.childName, Data(unprotectedContents, 'text/plain'))
        self.portal.registerChecker(AllowAnonymousAccess())
        self.credentialFactories.append(BasicCredentialFactory('example.com'))
        request = self.makeRequest([self.childName])
        child = getChildForRequest(self.wrapper, request)
        d = request.notifyFinish()

        def cbFinished(ignored):
            self.assertEqual(request.written, [unprotectedContents])
        d.addCallback(cbFinished)
        request.render(child)
        return d
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
class DigestAuthTests(RequestMixin, unittest.TestCase):
    """
    Digest authentication tests which use L{twisted.web.http.Request}.
    """

    def setUp(self):
        """
        Create a DigestCredentialFactory for testing
        """
        self.realm = b'test realm'
        self.algorithm = b'md5'
        self.credentialFactory = digest.DigestCredentialFactory(self.algorithm, self.realm)
        self.request = self.makeRequest()

    def test_decode(self):
        """
        L{digest.DigestCredentialFactory.decode} calls the C{decode} method on
        L{twisted.cred.digest.DigestCredentialFactory} with the HTTP method and
        host of the request.
        """
        host = b'169.254.0.1'
        method = b'GET'
        done = [False]
        response = object()

        def check(_response, _method, _host):
            self.assertEqual(response, _response)
            self.assertEqual(method, _method)
            self.assertEqual(host, _host)
            done[0] = True
        self.patch(self.credentialFactory.digest, 'decode', check)
        req = self.makeRequest(method, IPv4Address('TCP', host, 81))
        self.credentialFactory.decode(response, req)
        self.assertTrue(done[0])

    def test_interface(self):
        """
        L{DigestCredentialFactory} implements L{ICredentialFactory}.
        """
        self.assertTrue(verifyObject(ICredentialFactory, self.credentialFactory))

    def test_getChallenge(self):
        """
        The challenge issued by L{DigestCredentialFactory.getChallenge} must
        include C{'qop'}, C{'realm'}, C{'algorithm'}, C{'nonce'}, and
        C{'opaque'} keys.  The values for the C{'realm'} and C{'algorithm'}
        keys must match the values supplied to the factory's initializer.
        None of the values may have newlines in them.
        """
        challenge = self.credentialFactory.getChallenge(self.request)
        self.assertEqual(challenge['qop'], b'auth')
        self.assertEqual(challenge['realm'], b'test realm')
        self.assertEqual(challenge['algorithm'], b'md5')
        self.assertIn('nonce', challenge)
        self.assertIn('opaque', challenge)
        for v in challenge.values():
            self.assertNotIn(b'\n', v)

    def test_getChallengeWithoutClientIP(self):
        """
        L{DigestCredentialFactory.getChallenge} can issue a challenge even if
        the L{Request} it is passed returns L{None} from C{getClientIP}.
        """
        request = self.makeRequest(b'GET', None)
        challenge = self.credentialFactory.getChallenge(request)
        self.assertEqual(challenge['qop'], b'auth')
        self.assertEqual(challenge['realm'], b'test realm')
        self.assertEqual(challenge['algorithm'], b'md5')
        self.assertIn('nonce', challenge)
        self.assertIn('opaque', challenge)
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
class BasicAuthTestsMixin:
    """
    L{TestCase} mixin class which defines a number of tests for
    L{basic.BasicCredentialFactory}.  Because this mixin defines C{setUp}, it
    must be inherited before L{TestCase}.
    """

    def setUp(self):
        self.request = self.makeRequest()
        self.realm = b'foo'
        self.username = b'dreid'
        self.password = b'S3CuR1Ty'
        self.credentialFactory = basic.BasicCredentialFactory(self.realm)

    def makeRequest(self, method=b'GET', clientAddress=None):
        """
        Create a request object to be passed to
        L{basic.BasicCredentialFactory.decode} along with a response value.
        Override this in a subclass.
        """
        raise NotImplementedError(f'{self.__class__!r} did not implement makeRequest')

    def test_interface(self):
        """
        L{BasicCredentialFactory} implements L{ICredentialFactory}.
        """
        self.assertTrue(verifyObject(ICredentialFactory, self.credentialFactory))

    def test_usernamePassword(self):
        """
        L{basic.BasicCredentialFactory.decode} turns a base64-encoded response
        into a L{UsernamePassword} object with a password which reflects the
        one which was encoded in the response.
        """
        response = b64encode(b''.join([self.username, b':', self.password]))
        creds = self.credentialFactory.decode(response, self.request)
        self.assertTrue(IUsernamePassword.providedBy(creds))
        self.assertTrue(creds.checkPassword(self.password))
        self.assertFalse(creds.checkPassword(self.password + b'wrong'))

    def test_incorrectPadding(self):
        """
        L{basic.BasicCredentialFactory.decode} decodes a base64-encoded
        response with incorrect padding.
        """
        response = b64encode(b''.join([self.username, b':', self.password]))
        response = response.strip(b'=')
        creds = self.credentialFactory.decode(response, self.request)
        self.assertTrue(verifyObject(IUsernamePassword, creds))
        self.assertTrue(creds.checkPassword(self.password))

    def test_invalidEncoding(self):
        """
        L{basic.BasicCredentialFactory.decode} raises L{LoginFailed} if passed
        a response which is not base64-encoded.
        """
        response = b'x'
        self.assertRaises(error.LoginFailed, self.credentialFactory.decode, response, self.makeRequest())

    def test_invalidCredentials(self):
        """
        L{basic.BasicCredentialFactory.decode} raises L{LoginFailed} when
        passed a response which is not valid base64-encoded text.
        """
        response = b64encode(b'123abc+/')
        self.assertRaises(error.LoginFailed, self.credentialFactory.decode, response, self.makeRequest())
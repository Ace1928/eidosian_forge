import base64
import inspect
import re
from io import BytesIO
from typing import Any, List, Optional, Tuple, Type
from zope.interface import directlyProvides, implementer
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.error
import twisted.cred.portal
from twisted import cred
from twisted.cred.checkers import AllowAnonymousAccess, ICredentialsChecker
from twisted.cred.credentials import IAnonymous
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import address, defer, error, interfaces, protocol, reactor, task
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.mail import smtp
from twisted.mail._cred import LOGINCredentials
from twisted.protocols import basic, loopback
from twisted.python.util import LineLog
from twisted.trial.unittest import TestCase
class ESMTPAuthenticationTests(TestCase):

    def assertServerResponse(self, bytes, response):
        """
        Assert that when the given bytes are delivered to the ESMTP server
        instance, it responds with the indicated lines.

        @type bytes: str
        @type response: list of str
        """
        self.transport.clear()
        self.server.dataReceived(bytes)
        self.assertEqual(response, self.transport.value().splitlines())

    def assertServerAuthenticated(self, loginArgs, username=b'username', password=b'password'):
        """
        Assert that a login attempt has been made, that the credentials and
        interfaces passed to it are correct, and that when the login request
        is satisfied, a successful response is sent by the ESMTP server
        instance.

        @param loginArgs: A C{list} previously passed to L{portalFactory}.
        @param username: The login user.
        @param password: The login password.
        """
        d, credentials, mind, interfaces = loginArgs.pop()
        self.assertEqual(loginArgs, [])
        self.assertTrue(twisted.cred.credentials.IUsernamePassword.providedBy(credentials))
        self.assertEqual(credentials.username, username)
        self.assertTrue(credentials.checkPassword(password))
        self.assertIn(smtp.IMessageDeliveryFactory, interfaces)
        self.assertIn(smtp.IMessageDelivery, interfaces)
        d.callback((smtp.IMessageDeliveryFactory, None, lambda: None))
        self.assertEqual([b'235 Authentication successful.'], self.transport.value().splitlines())

    def setUp(self):
        """
        Create an ESMTP instance attached to a StringTransport.
        """
        self.server = smtp.ESMTP({b'LOGIN': LOGINCredentials})
        self.server.host = b'localhost'
        self.transport = StringTransport(peerAddress=address.IPv4Address('TCP', '127.0.0.1', 12345))
        self.server.makeConnection(self.transport)

    def tearDown(self):
        """
        Disconnect the ESMTP instance to clean up its timeout DelayedCall.
        """
        self.server.connectionLost(error.ConnectionDone())

    def portalFactory(self, loginList):

        class DummyPortal:

            def login(self, credentials, mind, *interfaces):
                d = defer.Deferred()
                loginList.append((d, credentials, mind, interfaces))
                return d
        return DummyPortal()

    def test_authenticationCapabilityAdvertised(self):
        """
        Test that AUTH is advertised to clients which issue an EHLO command.
        """
        self.transport.clear()
        self.server.dataReceived(b'EHLO\r\n')
        responseLines = self.transport.value().splitlines()
        self.assertEqual(responseLines[0], b'250-localhost Hello 127.0.0.1, nice to meet you')
        self.assertEqual(responseLines[1], b'250 AUTH LOGIN')
        self.assertEqual(len(responseLines), 2)

    def test_plainAuthentication(self):
        """
        Test that the LOGIN authentication mechanism can be used
        """
        loginArgs = []
        self.server.portal = self.portalFactory(loginArgs)
        self.server.dataReceived(b'EHLO\r\n')
        self.transport.clear()
        self.assertServerResponse(b'AUTH LOGIN\r\n', [b'334 ' + base64.b64encode(b'User Name\x00').strip()])
        self.assertServerResponse(base64.b64encode(b'username') + b'\r\n', [b'334 ' + base64.b64encode(b'Password\x00').strip()])
        self.assertServerResponse(base64.b64encode(b'password').strip() + b'\r\n', [])
        self.assertServerAuthenticated(loginArgs)

    def test_plainAuthenticationEmptyPassword(self):
        """
        Test that giving an empty password for plain auth succeeds.
        """
        loginArgs = []
        self.server.portal = self.portalFactory(loginArgs)
        self.server.dataReceived(b'EHLO\r\n')
        self.transport.clear()
        self.assertServerResponse(b'AUTH LOGIN\r\n', [b'334 ' + base64.b64encode(b'User Name\x00').strip()])
        self.assertServerResponse(base64.b64encode(b'username') + b'\r\n', [b'334 ' + base64.b64encode(b'Password\x00').strip()])
        self.assertServerResponse(b'\r\n', [])
        self.assertServerAuthenticated(loginArgs, password=b'')

    def test_plainAuthenticationInitialResponse(self):
        """
        The response to the first challenge may be included on the AUTH command
        line.  Test that this is also supported.
        """
        loginArgs = []
        self.server.portal = self.portalFactory(loginArgs)
        self.server.dataReceived(b'EHLO\r\n')
        self.transport.clear()
        self.assertServerResponse(b'AUTH LOGIN ' + base64.b64encode(b'username').strip() + b'\r\n', [b'334 ' + base64.b64encode(b'Password\x00').strip()])
        self.assertServerResponse(base64.b64encode(b'password').strip() + b'\r\n', [])
        self.assertServerAuthenticated(loginArgs)

    def test_abortAuthentication(self):
        """
        Test that a challenge/response sequence can be aborted by the client.
        """
        loginArgs = []
        self.server.portal = self.portalFactory(loginArgs)
        self.server.dataReceived(b'EHLO\r\n')
        self.server.dataReceived(b'AUTH LOGIN\r\n')
        self.assertServerResponse(b'*\r\n', [b'501 Authentication aborted'])

    def test_invalidBase64EncodedResponse(self):
        """
        Test that a response which is not properly Base64 encoded results in
        the appropriate error code.
        """
        loginArgs = []
        self.server.portal = self.portalFactory(loginArgs)
        self.server.dataReceived(b'EHLO\r\n')
        self.server.dataReceived(b'AUTH LOGIN\r\n')
        self.assertServerResponse(b'x\r\n', [b'501 Syntax error in parameters or arguments'])
        self.assertEqual(loginArgs, [])

    def test_invalidBase64EncodedInitialResponse(self):
        """
        Like L{test_invalidBase64EncodedResponse} but for the case of an
        initial response included with the C{AUTH} command.
        """
        loginArgs = []
        self.server.portal = self.portalFactory(loginArgs)
        self.server.dataReceived(b'EHLO\r\n')
        self.assertServerResponse(b'AUTH LOGIN x\r\n', [b'501 Syntax error in parameters or arguments'])
        self.assertEqual(loginArgs, [])

    def test_unexpectedLoginFailure(self):
        """
        If the L{Deferred} returned by L{Portal.login} fires with an
        exception of any type other than L{UnauthorizedLogin}, the exception
        is logged and the client is informed that the authentication attempt
        has failed.
        """
        loginArgs = []
        self.server.portal = self.portalFactory(loginArgs)
        self.server.dataReceived(b'EHLO\r\n')
        self.transport.clear()
        self.assertServerResponse(b'AUTH LOGIN ' + base64.b64encode(b'username').strip() + b'\r\n', [b'334 ' + base64.b64encode(b'Password\x00').strip()])
        self.assertServerResponse(base64.b64encode(b'password').strip() + b'\r\n', [])
        d, credentials, mind, interfaces = loginArgs.pop()
        d.errback(RuntimeError('Something wrong with the server'))
        self.assertEqual(b'451 Requested action aborted: local error in processing\r\n', self.transport.value())
        self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)
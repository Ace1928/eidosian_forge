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
class SMTPClientTests(TestCase, LoopbackMixin):
    """
    Tests for L{smtp.SMTPClient}.
    """

    def test_timeoutConnection(self):
        """
        L{smtp.SMTPClient.timeoutConnection} calls the C{sendError} hook with a
        fatal L{SMTPTimeoutError} with the current line log.
        """
        errors = []
        client = MySMTPClient()
        client.sendError = errors.append
        client.makeConnection(StringTransport())
        client.lineReceived(b'220 hello')
        client.timeoutConnection()
        self.assertIsInstance(errors[0], smtp.SMTPTimeoutError)
        self.assertTrue(errors[0].isFatal)
        self.assertEqual(bytes(errors[0]), b'Timeout waiting for SMTP server response\n<<< 220 hello\n>>> HELO foo.baz\n')
    expected_output = [b'HELO foo.baz', b'MAIL FROM:<moshez@foo.bar>', b'RCPT TO:<moshez@foo.bar>', b'DATA', b'Subject: hello', b'', b'Goodbye', b'.', b'RSET']

    def test_messages(self):
        """
        L{smtp.SMTPClient} sends I{HELO}, I{MAIL FROM}, I{RCPT TO}, and I{DATA}
        commands based on the return values of its C{getMailFrom},
        C{getMailTo}, and C{getMailData} methods.
        """
        client = MySMTPClient()
        server = FakeSMTPServer()
        d = self.loopback(server, client)
        d.addCallback(lambda x: self.assertEqual(server.buffer, self.expected_output))
        return d

    def test_transferError(self):
        """
        If there is an error while producing the message body to the
        connection, the C{sendError} callback is invoked.
        """
        client = MySMTPClient(('alice@example.com', ['bob@example.com'], BytesIO(b'foo')))
        transport = StringTransport()
        client.makeConnection(transport)
        client.dataReceived(b'220 Ok\r\n250 Ok\r\n250 Ok\r\n250 Ok\r\n354 Ok\r\n')
        self.assertNotIdentical(transport.producer, None)
        self.assertFalse(transport.streaming)
        transport.producer.stopProducing()
        self.assertIsInstance(client._error, Exception)

    def test_sendFatalError(self):
        """
        If L{smtp.SMTPClient.sendError} is called with an L{SMTPClientError}
        which is fatal, it disconnects its transport without writing anything
        more to it.
        """
        client = smtp.SMTPClient(None)
        transport = StringTransport()
        client.makeConnection(transport)
        client.sendError(smtp.SMTPClientError(123, 'foo', isFatal=True))
        self.assertEqual(transport.value(), b'')
        self.assertTrue(transport.disconnecting)

    def test_sendNonFatalError(self):
        """
        If L{smtp.SMTPClient.sendError} is called with an L{SMTPClientError}
        which is not fatal, it sends C{"QUIT"} and waits for the server to
        close the connection.
        """
        client = smtp.SMTPClient(None)
        transport = StringTransport()
        client.makeConnection(transport)
        client.sendError(smtp.SMTPClientError(123, 'foo', isFatal=False))
        self.assertEqual(transport.value(), b'QUIT\r\n')
        self.assertFalse(transport.disconnecting)

    def test_sendOtherError(self):
        """
        If L{smtp.SMTPClient.sendError} is called with an exception which is
        not an L{SMTPClientError}, it disconnects its transport without
        writing anything more to it.
        """
        client = smtp.SMTPClient(None)
        transport = StringTransport()
        client.makeConnection(transport)
        client.sendError(Exception('foo'))
        self.assertEqual(transport.value(), b'')
        self.assertTrue(transport.disconnecting)
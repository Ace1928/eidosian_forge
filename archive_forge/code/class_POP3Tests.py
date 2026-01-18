import base64
import hmac
import itertools
from collections import OrderedDict
from hashlib import md5
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.portal
import twisted.internet.protocol
import twisted.mail.pop3
import twisted.mail.protocols
from twisted import cred, internet, mail
from twisted.cred.credentials import IUsernameHashedPassword
from twisted.internet import defer
from twisted.internet.testing import LineSendingProtocol
from twisted.mail import pop3
from twisted.protocols import loopback
from twisted.python import failure
from twisted.trial import unittest, util
class POP3Tests(unittest.TestCase):
    """
    Tests for L{pop3.POP3}.
    """
    message = b'Subject: urgent\n\nSomeone set up us the bomb!\n'
    expectedOutput = b'+OK <moshez>\r\n+OK Authentication succeeded\r\n+OK \r\n1 0\r\n.\r\n+OK %d\r\nSubject: urgent\r\n\r\nSomeone set up us the bomb!\r\n.\r\n+OK \r\n' % (len(message),)

    def setUp(self):
        """
        Set up a POP3 server with virtual domain support.
        """
        self.factory = internet.protocol.Factory()
        self.factory.domains = {}
        self.factory.domains[b'baz.com'] = DummyDomain()
        self.factory.domains[b'baz.com'].addUser(b'hello')
        self.factory.domains[b'baz.com'].addMessage(b'hello', self.message)

    def test_messages(self):
        """
        Messages can be downloaded over a loopback TCP connection.
        """
        client = LineSendingProtocol([b'APOP hello@baz.com world', b'UIDL', b'RETR 1', b'QUIT'])
        server = MyVirtualPOP3()
        server.service = self.factory

        def check(ignored):
            output = b'\r\n'.join(client.response) + b'\r\n'
            self.assertEqual(output, self.expectedOutput)
        return loopback.loopbackTCP(server, client).addCallback(check)

    def test_loopback(self):
        """
        Messages can be downloaded over a loopback connection.
        """
        protocol = MyVirtualPOP3()
        protocol.service = self.factory
        clientProtocol = MyPOP3Downloader()

        def check(ignored):
            self.assertEqual(clientProtocol.message, self.message)
            protocol.connectionLost(failure.Failure(Exception('Test harness disconnect')))
        d = loopback.loopbackAsync(protocol, clientProtocol)
        return d.addCallback(check)
    test_loopback.suppress = [util.suppress(message='twisted.mail.pop3.POP3Client is deprecated')]

    def test_incorrectDomain(self):
        """
        Look up a user in a domain which this server does not support.
        """
        factory = internet.protocol.Factory()
        factory.domains = {}
        factory.domains[b'twistedmatrix.com'] = DummyDomain()
        server = MyVirtualPOP3()
        server.service = factory
        exc = self.assertRaises(pop3.POP3Error, server.authenticateUserAPOP, b'nobody@baz.com', b'password')
        self.assertEqual(exc.args[0], 'no such domain baz.com')
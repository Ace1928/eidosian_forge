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
class SMTPHelperTests(TestCase):

    def testMessageID(self):
        d = {}
        for i in range(1000):
            m = smtp.messageid('testcase')
            self.assertFalse(m in d)
            d[m] = None

    def testQuoteAddr(self):
        cases = [[b'user@host.name', b'<user@host.name>'], [b'"User Name" <user@host.name>', b'<user@host.name>'], [smtp.Address(b'someguy@someplace'), b'<someguy@someplace>'], [b'', b'<>'], [smtp.Address(b''), b'<>']]
        for c, e in cases:
            self.assertEqual(smtp.quoteaddr(c), e)

    def testUser(self):
        u = smtp.User(b'user@host', b'helo.host.name', None, None)
        self.assertEqual(str(u), 'user@host')

    def testXtextEncoding(self):
        cases = [('Hello world', b'Hello+20world'), ('Hello+world', b'Hello+2Bworld'), ('\x00\x01\x02\x03\x04\x05', b'+00+01+02+03+04+05'), ('e=mc2@example.com', b'e+3Dmc2@example.com')]
        for case, expected in cases:
            self.assertEqual(smtp.xtext_encode(case), (expected, len(case)))
            self.assertEqual(case.encode('xtext'), expected)
            self.assertEqual(smtp.xtext_decode(expected), (case, len(expected)))
            self.assertEqual(expected.decode('xtext'), case)

    def test_encodeWithErrors(self):
        """
        Specifying an error policy to C{unicode.encode} with the
        I{xtext} codec should produce the same result as not
        specifying the error policy.
        """
        text = 'Hello world'
        self.assertEqual(smtp.xtext_encode(text, 'strict'), (text.encode('xtext'), len(text)))
        self.assertEqual(text.encode('xtext', 'strict'), text.encode('xtext'))

    def test_decodeWithErrors(self):
        """
        Similar to L{test_encodeWithErrors}, but for C{bytes.decode}.
        """
        bytes = b'Hello world'
        self.assertEqual(smtp.xtext_decode(bytes, 'strict'), (bytes.decode('xtext'), len(bytes)))
        self.assertEqual(bytes.decode('xtext', 'strict'), bytes.decode('xtext'))
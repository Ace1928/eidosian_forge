import inspect
import sys
from typing import List
from unittest import skipIf
from zope.interface import directlyProvides
import twisted.mail._pop3client
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.mail.pop3 import (
from twisted.mail.test import pop3testserver
from twisted.protocols import basic, loopback
from twisted.python import log
from twisted.trial.unittest import TestCase
class POP3ClientLoginTests(TestCase):

    def testNegativeGreeting(self):
        p, t = setUp(greet=False)
        p.allowInsecureLogin = True
        d = p.login(b'username', b'password')
        p.dataReceived(b'-ERR Offline for maintenance\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'Offline for maintenance'))

    def testOkUser(self):
        p, t = setUp()
        d = p.user(b'username')
        self.assertEqual(t.value(), b'USER username\r\n')
        p.dataReceived(b'+OK send password\r\n')
        return d.addCallback(self.assertEqual, b'send password')

    def testBadUser(self):
        p, t = setUp()
        d = p.user(b'username')
        self.assertEqual(t.value(), b'USER username\r\n')
        p.dataReceived(b'-ERR account suspended\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'account suspended'))

    def testOkPass(self):
        p, t = setUp()
        d = p.password(b'password')
        self.assertEqual(t.value(), b'PASS password\r\n')
        p.dataReceived(b"+OK you're in!\r\n")
        return d.addCallback(self.assertEqual, b"you're in!")

    def testBadPass(self):
        p, t = setUp()
        d = p.password(b'password')
        self.assertEqual(t.value(), b'PASS password\r\n')
        p.dataReceived(b'-ERR go away\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'go away'))

    def testOkLogin(self):
        p, t = setUp()
        p.allowInsecureLogin = True
        d = p.login(b'username', b'password')
        self.assertEqual(t.value(), b'USER username\r\n')
        p.dataReceived(b'+OK go ahead\r\n')
        self.assertEqual(t.value(), b'USER username\r\nPASS password\r\n')
        p.dataReceived(b'+OK password accepted\r\n')
        return d.addCallback(self.assertEqual, b'password accepted')

    def testBadPasswordLogin(self):
        p, t = setUp()
        p.allowInsecureLogin = True
        d = p.login(b'username', b'password')
        self.assertEqual(t.value(), b'USER username\r\n')
        p.dataReceived(b'+OK waiting on you\r\n')
        self.assertEqual(t.value(), b'USER username\r\nPASS password\r\n')
        p.dataReceived(b'-ERR bogus login\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'bogus login'))

    def testBadUsernameLogin(self):
        p, t = setUp()
        p.allowInsecureLogin = True
        d = p.login(b'username', b'password')
        self.assertEqual(t.value(), b'USER username\r\n')
        p.dataReceived(b'-ERR bogus login\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'bogus login'))

    def testServerGreeting(self):
        p, t = setUp(greet=False)
        p.dataReceived(b'+OK lalala this has no challenge\r\n')
        self.assertEqual(p.serverChallenge, None)

    def testServerGreetingWithChallenge(self):
        p, t = setUp(greet=False)
        p.dataReceived(b'+OK <here is the challenge>\r\n')
        self.assertEqual(p.serverChallenge, b'<here is the challenge>')

    def testAPOP(self):
        p, t = setUp(greet=False)
        p.dataReceived(b'+OK <challenge string goes here>\r\n')
        d = p.login(b'username', b'password')
        self.assertEqual(t.value(), b'APOP username f34f1e464d0d7927607753129cabe39a\r\n')
        p.dataReceived(b'+OK Welcome!\r\n')
        return d.addCallback(self.assertEqual, b'Welcome!')

    def testInsecureLoginRaisesException(self):
        p, t = setUp(greet=False)
        p.dataReceived(b'+OK Howdy\r\n')
        d = p.login(b'username', b'password')
        self.assertFalse(t.value())
        return self.assertFailure(d, InsecureAuthenticationDisallowed)

    def testSSLTransportConsideredSecure(self):
        """
        If a server doesn't offer APOP but the transport is secured using
        SSL or TLS, a plaintext login should be allowed, not rejected with
        an InsecureAuthenticationDisallowed exception.
        """
        p, t = setUp(greet=False)
        directlyProvides(t, interfaces.ISSLTransport)
        p.dataReceived(b'+OK Howdy\r\n')
        d = p.login(b'username', b'password')
        self.assertEqual(t.value(), b'USER username\r\n')
        t.clear()
        p.dataReceived(b'+OK\r\n')
        self.assertEqual(t.value(), b'PASS password\r\n')
        p.dataReceived(b'+OK\r\n')
        return d
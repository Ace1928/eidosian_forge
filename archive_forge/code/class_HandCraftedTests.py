from __future__ import annotations
import base64
import codecs
import functools
import locale
import os
import uuid
from collections import OrderedDict
from io import BytesIO
from itertools import chain
from typing import Optional, Type
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, error, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.mail import imap4
from twisted.mail.imap4 import MessageSet
from twisted.mail.interfaces import (
from twisted.protocols import loopback
from twisted.python import failure, log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.trial.unittest import SynchronousTestCase, TestCase
class HandCraftedTests(IMAP4HelperMixin, TestCase):

    def testTrailingLiteral(self):
        transport = StringTransport()
        c = imap4.IMAP4Client()
        c.makeConnection(transport)
        c.lineReceived(b'* OK [IMAP4rev1]')

        def cbCheckTransport(ignored):
            self.assertEqual(transport.value().splitlines()[-1], b'0003 FETCH 1 (RFC822)')

        def cbSelect(ignored):
            d = c.fetchMessage('1')
            c.dataReceived(b'* 1 FETCH (RFC822 {10}\r\n0123456789\r\n RFC822.SIZE 10)\r\n')
            c.dataReceived(b'0003 OK FETCH\r\n')
            d.addCallback(cbCheckTransport)
            return d

        def cbLogin(ignored):
            d = c.select('inbox')
            c.lineReceived(b'0002 OK SELECT')
            d.addCallback(cbSelect)
            return d
        d = c.login(b'blah', b'blah')
        c.dataReceived(b'0001 OK LOGIN\r\n')
        d.addCallback(cbLogin)
        return d

    def test_fragmentedStringLiterals(self):
        """
        String literals whose data is not immediately available are
        parsed.
        """
        self.server.checker.addUser(b'testuser', b'password-test')
        transport = StringTransport()
        self.server.makeConnection(transport)
        transport.clear()
        self.server.dataReceived(b'01 LOGIN {8}\r\n')
        self.assertEqual(transport.value(), b'+ Ready for 8 octets of text\r\n')
        transport.clear()
        self.server.dataReceived(b'testuser {13}\r\n')
        self.assertEqual(transport.value(), b'+ Ready for 13 octets of text\r\n')
        transport.clear()
        self.server.dataReceived(b'password')
        self.assertNot(transport.value())
        self.server.dataReceived(b'-test\r\n')
        self.assertEqual(transport.value(), b'01 OK LOGIN succeeded\r\n')
        self.assertEqual(self.server.state, 'auth')
        self.server.connectionLost(error.ConnectionDone('Connection done.'))

    def test_emptyStringLiteral(self):
        """
        Empty string literals are parsed.
        """
        self.server.checker.users = {b'': b''}
        transport = StringTransport()
        self.server.makeConnection(transport)
        transport.clear()
        self.server.dataReceived(b'01 LOGIN {0}\r\n')
        self.assertEqual(transport.value(), b'+ Ready for 0 octets of text\r\n')
        transport.clear()
        self.server.dataReceived(b'{0}\r\n')
        self.assertEqual(transport.value(), b'01 OK LOGIN succeeded\r\n')
        self.assertEqual(self.server.state, 'auth')
        self.server.connectionLost(error.ConnectionDone('Connection done.'))

    def test_unsolicitedResponseMixedWithSolicitedResponse(self):
        """
        If unsolicited data is received along with solicited data in the
        response to a I{FETCH} command issued by L{IMAP4Client.fetchSpecific},
        the unsolicited data is passed to the appropriate callback and not
        included in the result with which the L{Deferred} returned by
        L{IMAP4Client.fetchSpecific} fires.
        """
        transport = StringTransport()
        c = StillSimplerClient()
        c.makeConnection(transport)
        c.lineReceived(b'* OK [IMAP4rev1]')

        def login():
            d = c.login(b'blah', b'blah')
            c.dataReceived(b'0001 OK LOGIN\r\n')
            return d

        def select():
            d = c.select('inbox')
            c.lineReceived(b'0002 OK SELECT')
            return d

        def fetch():
            d = c.fetchSpecific('1:*', headerType='HEADER.FIELDS', headerArgs=['SUBJECT'])
            c.dataReceived(b'* 1 FETCH (BODY[HEADER.FIELDS ("SUBJECT")] {38}\r\n')
            c.dataReceived(b'Subject: Suprise for your woman...\r\n')
            c.dataReceived(b'\r\n')
            c.dataReceived(b')\r\n')
            c.dataReceived(b'* 1 FETCH (FLAGS (\\Seen))\r\n')
            c.dataReceived(b'* 2 FETCH (BODY[HEADER.FIELDS ("SUBJECT")] {75}\r\n')
            c.dataReceived(b'Subject: What you been doing. Order your meds here . ,. handcuff madsen\r\n')
            c.dataReceived(b'\r\n')
            c.dataReceived(b')\r\n')
            c.dataReceived(b'0003 OK FETCH completed\r\n')
            return d

        def test(res):
            self.assertEqual(transport.value().splitlines()[-1], b'0003 FETCH 1:* BODY[HEADER.FIELDS (SUBJECT)]')
            self.assertEqual(res, {1: [['BODY', ['HEADER.FIELDS', ['SUBJECT']], 'Subject: Suprise for your woman...\r\n\r\n']], 2: [['BODY', ['HEADER.FIELDS', ['SUBJECT']], 'Subject: What you been doing. Order your meds here . ,. handcuff madsen\r\n\r\n']]})
            self.assertEqual(c.flags, {1: ['\\Seen']})
        return login().addCallback(strip(select)).addCallback(strip(fetch)).addCallback(test)

    def test_literalWithoutPrecedingWhitespace(self):
        """
        Literals should be recognized even when they are not preceded by
        whitespace.
        """
        transport = StringTransport()
        protocol = imap4.IMAP4Client()
        protocol.makeConnection(transport)
        protocol.lineReceived(b'* OK [IMAP4rev1]')

        def login():
            d = protocol.login(b'blah', b'blah')
            protocol.dataReceived(b'0001 OK LOGIN\r\n')
            return d

        def select():
            d = protocol.select(b'inbox')
            protocol.lineReceived(b'0002 OK SELECT')
            return d

        def fetch():
            d = protocol.fetchSpecific('1:*', headerType='HEADER.FIELDS', headerArgs=['SUBJECT'])
            protocol.dataReceived(b'* 1 FETCH (BODY[HEADER.FIELDS ({7}\r\nSUBJECT)] "Hello")\r\n')
            protocol.dataReceived(b'0003 OK FETCH completed\r\n')
            return d

        def test(result):
            self.assertEqual(transport.value().splitlines()[-1], b'0003 FETCH 1:* BODY[HEADER.FIELDS (SUBJECT)]')
            self.assertEqual(result, {1: [['BODY', ['HEADER.FIELDS', ['SUBJECT']], 'Hello']]})
        d = login()
        d.addCallback(strip(select))
        d.addCallback(strip(fetch))
        d.addCallback(test)
        return d

    def test_nonIntegerLiteralLength(self):
        """
        If the server sends a literal length which cannot be parsed as an
        integer, L{IMAP4Client.lineReceived} should cause the protocol to be
        disconnected by raising L{imap4.IllegalServerResponse}.
        """
        transport = StringTransport()
        protocol = imap4.IMAP4Client()
        protocol.makeConnection(transport)
        protocol.lineReceived(b'* OK [IMAP4rev1]')

        def login():
            d = protocol.login(b'blah', b'blah')
            protocol.dataReceived(b'0001 OK LOGIN\r\n')
            return d

        def select():
            d = protocol.select('inbox')
            protocol.lineReceived(b'0002 OK SELECT')
            return d

        def fetch():
            protocol.fetchSpecific('1:*', headerType='HEADER.FIELDS', headerArgs=['SUBJECT'])
            self.assertEqual(transport.value().splitlines()[-1], b'0003 FETCH 1:* BODY[HEADER.FIELDS (SUBJECT)]')
            self.assertRaises(imap4.IllegalServerResponse, protocol.dataReceived, b'* 1 FETCH {xyz}\r\n...')
        d = login()
        d.addCallback(strip(select))
        d.addCallback(strip(fetch))
        return d

    def test_flagsChangedInsideFetchSpecificResponse(self):
        """
        Any unrequested flag information received along with other requested
        information in an untagged I{FETCH} received in response to a request
        issued with L{IMAP4Client.fetchSpecific} is passed to the
        C{flagsChanged} callback.
        """
        transport = StringTransport()
        c = StillSimplerClient()
        c.makeConnection(transport)
        c.lineReceived(b'* OK [IMAP4rev1]')

        def login():
            d = c.login(b'blah', b'blah')
            c.dataReceived(b'0001 OK LOGIN\r\n')
            return d

        def select():
            d = c.select('inbox')
            c.lineReceived(b'0002 OK SELECT')
            return d

        def fetch():
            d = c.fetchSpecific(b'1:*', headerType='HEADER.FIELDS', headerArgs=['SUBJECT'])
            c.dataReceived(b'* 1 FETCH (BODY[HEADER.FIELDS ("SUBJECT")] {22}\r\n')
            c.dataReceived(b'Subject: subject one\r\n')
            c.dataReceived(b' FLAGS (\\Recent))\r\n')
            c.dataReceived(b'* 2 FETCH (FLAGS (\\Seen) BODY[HEADER.FIELDS ("SUBJECT")] {22}\r\n')
            c.dataReceived(b'Subject: subject two\r\n')
            c.dataReceived(b')\r\n')
            c.dataReceived(b'0003 OK FETCH completed\r\n')
            return d

        def test(res):
            self.assertEqual(res, {1: [['BODY', ['HEADER.FIELDS', ['SUBJECT']], 'Subject: subject one\r\n']], 2: [['BODY', ['HEADER.FIELDS', ['SUBJECT']], 'Subject: subject two\r\n']]})
            self.assertEqual(c.flags, {1: ['\\Recent'], 2: ['\\Seen']})
        return login().addCallback(strip(select)).addCallback(strip(fetch)).addCallback(test)

    def test_flagsChangedInsideFetchMessageResponse(self):
        """
        Any unrequested flag information received along with other requested
        information in an untagged I{FETCH} received in response to a request
        issued with L{IMAP4Client.fetchMessage} is passed to the
        C{flagsChanged} callback.
        """
        transport = StringTransport()
        c = StillSimplerClient()
        c.makeConnection(transport)
        c.lineReceived(b'* OK [IMAP4rev1]')

        def login():
            d = c.login(b'blah', b'blah')
            c.dataReceived(b'0001 OK LOGIN\r\n')
            return d

        def select():
            d = c.select('inbox')
            c.lineReceived(b'0002 OK SELECT')
            return d

        def fetch():
            d = c.fetchMessage('1:*')
            c.dataReceived(b'* 1 FETCH (RFC822 {24}\r\n')
            c.dataReceived(b'Subject: first subject\r\n')
            c.dataReceived(b' FLAGS (\\Seen))\r\n')
            c.dataReceived(b'* 2 FETCH (FLAGS (\\Recent \\Seen) RFC822 {25}\r\n')
            c.dataReceived(b'Subject: second subject\r\n')
            c.dataReceived(b')\r\n')
            c.dataReceived(b'0003 OK FETCH completed\r\n')
            return d

        def test(res):
            self.assertEqual(transport.value().splitlines()[-1], b'0003 FETCH 1:* (RFC822)')
            self.assertEqual(res, {1: {'RFC822': 'Subject: first subject\r\n'}, 2: {'RFC822': 'Subject: second subject\r\n'}})
            self.assertEqual(c.flags, {1: ['\\Seen'], 2: ['\\Recent', '\\Seen']})
        return login().addCallback(strip(select)).addCallback(strip(fetch)).addCallback(test)

    def test_authenticationChallengeDecodingException(self):
        """
        When decoding a base64 encoded authentication message from the server,
        decoding errors are logged and then the client closes the connection.
        """
        transport = StringTransportWithDisconnection()
        protocol = imap4.IMAP4Client()
        transport.protocol = protocol
        protocol.makeConnection(transport)
        protocol.lineReceived(b'* OK [CAPABILITY IMAP4rev1 IDLE NAMESPACE AUTH=CRAM-MD5] Twisted IMAP4rev1 Ready')
        cAuth = imap4.CramMD5ClientAuthenticator(b'testuser')
        protocol.registerAuthenticator(cAuth)
        d = protocol.authenticate('secret')
        self.assertFailure(d, error.ConnectionDone)
        protocol.dataReceived(b'+ Something bad! and bad\r\n')
        logged = self.flushLoggedErrors(imap4.IllegalServerResponse)
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0].value.args[0], b'Something bad! and bad')
        return d
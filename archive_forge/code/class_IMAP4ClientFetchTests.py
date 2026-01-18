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
class IMAP4ClientFetchTests(PreauthIMAP4ClientMixin, SynchronousTestCase):
    """
    Tests for the L{IMAP4Client.fetch} method.

    See RFC 3501, section 6.4.5.
    """

    def test_fetchUID(self):
        """
        L{IMAP4Client.fetchUID} sends the I{FETCH UID} command and returns a
        L{Deferred} which fires with a C{dict} mapping message sequence numbers
        to C{dict}s mapping C{'UID'} to that message's I{UID} in the server's
        response.
        """
        d = self.client.fetchUID('1:7')
        self.assertEqual(self.transport.value(), b'0001 FETCH 1:7 (UID)\r\n')
        self.client.lineReceived(b'* 2 FETCH (UID 22)')
        self.client.lineReceived(b'* 3 FETCH (UID 23)')
        self.client.lineReceived(b'* 4 FETCH (UID 24)')
        self.client.lineReceived(b'* 5 FETCH (UID 25)')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.assertEqual(self.successResultOf(d), {2: {'UID': '22'}, 3: {'UID': '23'}, 4: {'UID': '24'}, 5: {'UID': '25'}})

    def test_fetchUIDNonIntegerFound(self):
        """
        If the server responds with a non-integer where a message sequence
        number is expected, the L{Deferred} returned by L{IMAP4Client.fetchUID}
        fails with L{IllegalServerResponse}.
        """
        d = self.client.fetchUID('1')
        self.assertEqual(self.transport.value(), b'0001 FETCH 1 (UID)\r\n')
        self.client.lineReceived(b'* foo FETCH (UID 22)')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.failureResultOf(d, imap4.IllegalServerResponse)

    def test_incompleteFetchUIDResponse(self):
        """
        If the server responds with an incomplete I{FETCH} response line, the
        L{Deferred} returned by L{IMAP4Client.fetchUID} fails with
        L{IllegalServerResponse}.
        """
        d = self.client.fetchUID('1:7')
        self.assertEqual(self.transport.value(), b'0001 FETCH 1:7 (UID)\r\n')
        self.client.lineReceived(b'* 2 FETCH (UID 22)')
        self.client.lineReceived(b'* 3 FETCH (UID)')
        self.client.lineReceived(b'* 4 FETCH (UID 24)')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.failureResultOf(d, imap4.IllegalServerResponse)

    def test_fetchBody(self):
        """
        L{IMAP4Client.fetchBody} sends the I{FETCH BODY} command and returns a
        L{Deferred} which fires with a C{dict} mapping message sequence numbers
        to C{dict}s mapping C{'RFC822.TEXT'} to that message's body as given in
        the server's response.
        """
        d = self.client.fetchBody('3')
        self.assertEqual(self.transport.value(), b'0001 FETCH 3 (RFC822.TEXT)\r\n')
        self.client.lineReceived(b'* 3 FETCH (RFC822.TEXT "Message text")')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.assertEqual(self.successResultOf(d), {3: {'RFC822.TEXT': 'Message text'}})

    def test_fetchSpecific(self):
        """
        L{IMAP4Client.fetchSpecific} sends the I{BODY[]} command if no
        parameters beyond the message set to retrieve are given.  It returns a
        L{Deferred} which fires with a C{dict} mapping message sequence numbers
        to C{list}s of corresponding message data given by the server's
        response.
        """
        d = self.client.fetchSpecific('7')
        self.assertEqual(self.transport.value(), b'0001 FETCH 7 BODY[]\r\n')
        self.client.lineReceived(b'* 7 FETCH (BODY[] "Some body")')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.assertEqual(self.successResultOf(d), {7: [['BODY', [], 'Some body']]})

    def test_fetchSpecificPeek(self):
        """
        L{IMAP4Client.fetchSpecific} issues a I{BODY.PEEK[]} command if passed
        C{True} for the C{peek} parameter.
        """
        d = self.client.fetchSpecific('6', peek=True)
        self.assertEqual(self.transport.value(), b'0001 FETCH 6 BODY.PEEK[]\r\n')
        self.client.lineReceived(b'* 6 FETCH (BODY[] "Some body")')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.assertEqual(self.successResultOf(d), {6: [['BODY', [], 'Some body']]})

    def test_fetchSpecificNumbered(self):
        """
        L{IMAP4Client.fetchSpecific}, when passed a sequence for
        C{headerNumber}, sends the I{BODY[N.M]} command.  It returns a
        L{Deferred} which fires with a C{dict} mapping message sequence numbers
        to C{list}s of corresponding message data given by the server's
        response.
        """
        d = self.client.fetchSpecific('7', headerNumber=(1, 2, 3))
        self.assertEqual(self.transport.value(), b'0001 FETCH 7 BODY[1.2.3]\r\n')
        self.client.lineReceived(b'* 7 FETCH (BODY[1.2.3] "Some body")')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.assertEqual(self.successResultOf(d), {7: [['BODY', ['1.2.3'], 'Some body']]})

    def test_fetchSpecificText(self):
        """
        L{IMAP4Client.fetchSpecific}, when passed C{'TEXT'} for C{headerType},
        sends the I{BODY[TEXT]} command.  It returns a L{Deferred} which fires
        with a C{dict} mapping message sequence numbers to C{list}s of
        corresponding message data given by the server's response.
        """
        d = self.client.fetchSpecific('8', headerType='TEXT')
        self.assertEqual(self.transport.value(), b'0001 FETCH 8 BODY[TEXT]\r\n')
        self.client.lineReceived(b'* 8 FETCH (BODY[TEXT] "Some body")')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.assertEqual(self.successResultOf(d), {8: [['BODY', ['TEXT'], 'Some body']]})

    def test_fetchSpecificNumberedText(self):
        """
        If passed a value for the C{headerNumber} parameter and C{'TEXT'} for
        the C{headerType} parameter, L{IMAP4Client.fetchSpecific} sends a
        I{BODY[number.TEXT]} request and returns a L{Deferred} which fires with
        a C{dict} mapping message sequence numbers to C{list}s of message data
        given by the server's response.
        """
        d = self.client.fetchSpecific('4', headerType='TEXT', headerNumber=7)
        self.assertEqual(self.transport.value(), b'0001 FETCH 4 BODY[7.TEXT]\r\n')
        self.client.lineReceived(b'* 4 FETCH (BODY[7.TEXT] "Some body")')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.assertEqual(self.successResultOf(d), {4: [['BODY', ['7.TEXT'], 'Some body']]})

    def test_incompleteFetchSpecificTextResponse(self):
        """
        If the server responds to a I{BODY[TEXT]} request with a I{FETCH} line
        which is truncated after the I{BODY[TEXT]} tokens, the L{Deferred}
        returned by L{IMAP4Client.fetchUID} fails with
        L{IllegalServerResponse}.
        """
        d = self.client.fetchSpecific('8', headerType='TEXT')
        self.assertEqual(self.transport.value(), b'0001 FETCH 8 BODY[TEXT]\r\n')
        self.client.lineReceived(b'* 8 FETCH (BODY[TEXT])')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.failureResultOf(d, imap4.IllegalServerResponse)

    def test_fetchSpecificMIME(self):
        """
        L{IMAP4Client.fetchSpecific}, when passed C{'MIME'} for C{headerType},
        sends the I{BODY[MIME]} command.  It returns a L{Deferred} which fires
        with a C{dict} mapping message sequence numbers to C{list}s of
        corresponding message data given by the server's response.
        """
        d = self.client.fetchSpecific('8', headerType='MIME')
        self.assertEqual(self.transport.value(), b'0001 FETCH 8 BODY[MIME]\r\n')
        self.client.lineReceived(b'* 8 FETCH (BODY[MIME] "Some body")')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.assertEqual(self.successResultOf(d), {8: [['BODY', ['MIME'], 'Some body']]})

    def test_fetchSpecificPartial(self):
        """
        L{IMAP4Client.fetchSpecific}, when passed C{offset} and C{length},
        sends a partial content request (like I{BODY[TEXT]<offset.length>}).
        It returns a L{Deferred} which fires with a C{dict} mapping message
        sequence numbers to C{list}s of corresponding message data given by the
        server's response.
        """
        d = self.client.fetchSpecific('9', headerType='TEXT', offset=17, length=3)
        self.assertEqual(self.transport.value(), b'0001 FETCH 9 BODY[TEXT]<17.3>\r\n')
        self.client.lineReceived(b'* 9 FETCH (BODY[TEXT]<17> "foo")')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.assertEqual(self.successResultOf(d), {9: [['BODY', ['TEXT'], '<17>', 'foo']]})

    def test_incompleteFetchSpecificPartialResponse(self):
        """
        If the server responds to a I{BODY[TEXT]} request with a I{FETCH} line
        which is truncated after the I{BODY[TEXT]<offset>} tokens, the
        L{Deferred} returned by L{IMAP4Client.fetchUID} fails with
        L{IllegalServerResponse}.
        """
        d = self.client.fetchSpecific('8', headerType='TEXT')
        self.assertEqual(self.transport.value(), b'0001 FETCH 8 BODY[TEXT]\r\n')
        self.client.lineReceived(b'* 8 FETCH (BODY[TEXT]<17>)')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.failureResultOf(d, imap4.IllegalServerResponse)

    def test_fetchSpecificHTML(self):
        """
        If the body of a message begins with I{<} and ends with I{>} (as,
        for example, HTML bodies typically will), this is still interpreted
        as the body by L{IMAP4Client.fetchSpecific} (and particularly, not
        as a length indicator for a response to a request for a partial
        body).
        """
        d = self.client.fetchSpecific('7')
        self.assertEqual(self.transport.value(), b'0001 FETCH 7 BODY[]\r\n')
        self.client.lineReceived(b'* 7 FETCH (BODY[] "<html>test</html>")')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.assertEqual(self.successResultOf(d), {7: [['BODY', [], '<html>test</html>']]})

    def assertFetchSpecificFieldsWithEmptyList(self, section):
        """
        Assert that the provided C{BODY} section, when invoked with no
        arguments, produces an empty list, and that it returns a
        L{Deferred} which fires with a C{dict} mapping message
        sequence numbers to C{list}s of corresponding message data
        given by the server's response.

        @param section: The C{BODY} section to test: either
            C{'HEADER.FIELDS'} or C{'HEADER.FIELDS.NOT'}
        @type section: L{str}
        """
        d = self.client.fetchSpecific('10', headerType=section)
        self.assertEqual(self.transport.value(), b'0001 FETCH 10 BODY[' + section.encode('ascii') + b' ()]\r\n')
        self.client.lineReceived(b'* 10 FETCH (BODY[' + section.encode('ascii') + b' ()] "")')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.assertEqual(self.successResultOf(d), {10: [['BODY', [section, []], '']]})

    def test_fetchSpecificHeaderFieldsWithoutHeaders(self):
        """
        L{IMAP4Client.fetchSpecific}, when passed C{'HEADER.FIELDS'}
        for C{headerType} but no C{headerArgs}, sends the
        I{BODY[HEADER.FIELDS]} command with no arguments.  It returns
        a L{Deferred} which fires with a C{dict} mapping message
        sequence numbers to C{list}s of corresponding message data
        given by the server's response.
        """
        self.assertFetchSpecificFieldsWithEmptyList('HEADER.FIELDS')

    def test_fetchSpecificHeaderFieldsNotWithoutHeaders(self):
        """
        L{IMAP4Client.fetchSpecific}, when passed
        C{'HEADER.FIELDS.NOT'} for C{headerType} but no C{headerArgs},
        sends the I{BODY[HEADER.FIELDS.NOT]} command with no
        arguments.  It returns a L{Deferred} which fires with a
        C{dict} mapping message sequence numbers to C{list}s of
        corresponding message data given by the server's response.
        """
        self.assertFetchSpecificFieldsWithEmptyList('HEADER.FIELDS.NOT')

    def test_fetchSpecificHeader(self):
        """
        L{IMAP4Client.fetchSpecific}, when passed C{'HEADER'} for
        C{headerType}, sends the I{BODY[HEADER]} command.  It returns
        a L{Deferred} which fires with a C{dict} mapping message
        sequence numbers to C{list}s of corresponding message data
        given by the server's response.
        """
        d = self.client.fetchSpecific('11', headerType='HEADER')
        self.assertEqual(self.transport.value(), b'0001 FETCH 11 BODY[HEADER]\r\n')
        self.client.lineReceived(b'* 11 FETCH (BODY[HEADER] "From: someone@localhost\r\nSubject: Some subject")')
        self.client.lineReceived(b'0001 OK FETCH completed')
        self.assertEqual(self.successResultOf(d), {11: [['BODY', ['HEADER'], 'From: someone@localhost\r\nSubject: Some subject']]})
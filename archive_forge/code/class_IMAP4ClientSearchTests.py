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
class IMAP4ClientSearchTests(PreauthIMAP4ClientMixin, SynchronousTestCase):
    """
    Tests for the L{IMAP4Client.search} method.

    An example of usage of the SEARCH command from RFC 3501, section 6.4.4::

        C: A282 SEARCH FLAGGED SINCE 1-Feb-1994 NOT FROM "Smith"
        S: * SEARCH 2 84 882
        S: A282 OK SEARCH completed
        C: A283 SEARCH TEXT "string not in mailbox"
        S: * SEARCH
        S: A283 OK SEARCH completed
        C: A284 SEARCH CHARSET UTF-8 TEXT {6}
        C: XXXXXX
        S: * SEARCH 43
        S: A284 OK SEARCH completed
    """

    def _search(self):
        d = self.client.search(imap4.Query(text='ABCDEF'))
        self.assertEqual(self.transport.value(), b'0001 SEARCH (TEXT "ABCDEF")\r\n')
        return d

    def _response(self, messageNumbers):
        self.client.lineReceived(b'* SEARCH ' + networkString(' '.join(map(str, messageNumbers))))
        self.client.lineReceived(b'0001 OK SEARCH completed')

    def test_search(self):
        """
        L{IMAP4Client.search} sends the I{SEARCH} command and returns a
        L{Deferred} which fires with a C{list} of message sequence numbers
        given by the server's response.
        """
        d = self._search()
        self._response([2, 5, 10])
        self.assertEqual(self.successResultOf(d), [2, 5, 10])

    def test_nonIntegerFound(self):
        """
        If the server responds with a non-integer where a message sequence
        number is expected, the L{Deferred} returned by L{IMAP4Client.search}
        fails with L{IllegalServerResponse}.
        """
        d = self._search()
        self._response([2, 'foo', 10])
        self.failureResultOf(d, imap4.IllegalServerResponse)
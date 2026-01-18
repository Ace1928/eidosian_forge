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
class IMAP4ClientExpungeTests(PreauthIMAP4ClientMixin, SynchronousTestCase):
    """
    Tests for the L{IMAP4Client.expunge} method.

    An example of usage of the EXPUNGE command from RFC 3501, section 6.4.3::

        C: A202 EXPUNGE
        S: * 3 EXPUNGE
        S: * 3 EXPUNGE
        S: * 5 EXPUNGE
        S: * 8 EXPUNGE
        S: A202 OK EXPUNGE completed
    """

    def _expunge(self):
        d = self.client.expunge()
        self.assertEqual(self.transport.value(), b'0001 EXPUNGE\r\n')
        self.transport.clear()
        return d

    def _response(self, sequenceNumbers):
        for number in sequenceNumbers:
            self.client.lineReceived(networkString(f'* {number} EXPUNGE'))
        self.client.lineReceived(b'0001 OK EXPUNGE COMPLETED')

    def test_expunge(self):
        """
        L{IMAP4Client.expunge} sends the I{EXPUNGE} command and returns a
        L{Deferred} which fires with a C{list} of message sequence numbers
        given by the server's response.
        """
        d = self._expunge()
        self._response([3, 3, 5, 8])
        self.assertEqual(self.successResultOf(d), [3, 3, 5, 8])

    def test_nonIntegerExpunged(self):
        """
        If the server responds with a non-integer where a message sequence
        number is expected, the L{Deferred} returned by L{IMAP4Client.expunge}
        fails with L{IllegalServerResponse}.
        """
        d = self._expunge()
        self._response([3, 3, 'foo', 8])
        self.failureResultOf(d, imap4.IllegalServerResponse)
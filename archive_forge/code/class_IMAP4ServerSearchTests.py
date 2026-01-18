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
class IMAP4ServerSearchTests(IMAP4HelperMixin, TestCase):
    """
    Tests for the behavior of the search_* functions in L{imap4.IMAP4Server}.
    """

    def setUp(self):
        IMAP4HelperMixin.setUp(self)
        self.earlierQuery = ['10-Dec-2009']
        self.sameDateQuery = ['13-Dec-2009']
        self.laterQuery = ['16-Dec-2009']
        self.seq = 0
        self.msg = FakeyMessage({'date': 'Mon, 13 Dec 2009 21:25:10 GMT'}, [], '13 Dec 2009 00:00:00 GMT', '', 1234, None)

    def test_searchSentBefore(self):
        """
        L{imap4.IMAP4Server.search_SENTBEFORE} returns True if the message date
        is earlier than the query date.
        """
        self.assertFalse(self.server.search_SENTBEFORE(self.earlierQuery, self.seq, self.msg))
        self.assertTrue(self.server.search_SENTBEFORE(self.laterQuery, self.seq, self.msg))

    def test_searchWildcard(self):
        """
        L{imap4.IMAP4Server.search_UID} returns True if the message UID is in
        the search range.
        """
        self.assertFalse(self.server.search_UID([b'2:3'], self.seq, self.msg, (1, 1234)))
        self.assertTrue(self.server.search_UID([b'2:*'], self.seq, self.msg, (1, 1234)))
        self.assertTrue(self.server.search_UID([b'*'], self.seq, self.msg, (1, 1234)))

    def test_searchWildcardHigh(self):
        """
        L{imap4.IMAP4Server.search_UID} should return True if there is a
        wildcard, because a wildcard means "highest UID in the mailbox".
        """
        self.assertTrue(self.server.search_UID([b'1235:*'], self.seq, self.msg, (1234, 1)))

    def test_reversedSearchTerms(self):
        """
        L{imap4.IMAP4Server.search_SENTON} returns True if the message date is
        the same as the query date.
        """
        msgset = imap4.parseIdList(b'4:2')
        self.assertEqual(list(msgset), [2, 3, 4])

    def test_searchSentOn(self):
        """
        L{imap4.IMAP4Server.search_SENTON} returns True if the message date is
        the same as the query date.
        """
        self.assertFalse(self.server.search_SENTON(self.earlierQuery, self.seq, self.msg))
        self.assertTrue(self.server.search_SENTON(self.sameDateQuery, self.seq, self.msg))
        self.assertFalse(self.server.search_SENTON(self.laterQuery, self.seq, self.msg))

    def test_searchSentSince(self):
        """
        L{imap4.IMAP4Server.search_SENTSINCE} returns True if the message date
        is later than the query date.
        """
        self.assertTrue(self.server.search_SENTSINCE(self.earlierQuery, self.seq, self.msg))
        self.assertFalse(self.server.search_SENTSINCE(self.laterQuery, self.seq, self.msg))

    def test_searchOr(self):
        """
        L{imap4.IMAP4Server.search_OR} returns true if either of the two
        expressions supplied to it returns true and returns false if neither
        does.
        """
        self.assertTrue(self.server.search_OR(['SENTSINCE'] + self.earlierQuery + ['SENTSINCE'] + self.laterQuery, self.seq, self.msg, (None, None)))
        self.assertTrue(self.server.search_OR(['SENTSINCE'] + self.laterQuery + ['SENTSINCE'] + self.earlierQuery, self.seq, self.msg, (None, None)))
        self.assertFalse(self.server.search_OR(['SENTON'] + self.laterQuery + ['SENTSINCE'] + self.laterQuery, self.seq, self.msg, (None, None)))

    def test_searchNot(self):
        """
        L{imap4.IMAP4Server.search_NOT} returns the negation of the result
        of the expression supplied to it.
        """
        self.assertFalse(self.server.search_NOT(['SENTSINCE'] + self.earlierQuery, self.seq, self.msg, (None, None)))
        self.assertTrue(self.server.search_NOT(['SENTON'] + self.laterQuery, self.seq, self.msg, (None, None)))

    def test_searchBefore(self):
        """
        L{imap4.IMAP4Server.search_BEFORE} returns True if the
        internal message date is before the query date.
        """
        self.assertFalse(self.server.search_BEFORE(self.earlierQuery, self.seq, self.msg))
        self.assertFalse(self.server.search_BEFORE(self.sameDateQuery, self.seq, self.msg))
        self.assertTrue(self.server.search_BEFORE(self.laterQuery, self.seq, self.msg))

    def test_searchOn(self):
        """
        L{imap4.IMAP4Server.search_ON} returns True if the
        internal message date is the same as the query date.
        """
        self.assertFalse(self.server.search_ON(self.earlierQuery, self.seq, self.msg))
        self.assertFalse(self.server.search_ON(self.sameDateQuery, self.seq, self.msg))
        self.assertFalse(self.server.search_ON(self.laterQuery, self.seq, self.msg))

    def test_searchSince(self):
        """
        L{imap4.IMAP4Server.search_SINCE} returns True if the
        internal message date is greater than the query date.
        """
        self.assertTrue(self.server.search_SINCE(self.earlierQuery, self.seq, self.msg))
        self.assertTrue(self.server.search_SINCE(self.sameDateQuery, self.seq, self.msg))
        self.assertFalse(self.server.search_SINCE(self.laterQuery, self.seq, self.msg))
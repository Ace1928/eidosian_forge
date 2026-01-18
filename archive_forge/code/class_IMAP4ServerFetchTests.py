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
class IMAP4ServerFetchTests(TestCase):
    """
    This test case is for the FETCH tests that require
    a C{StringTransport}.
    """

    def setUp(self):
        self.transport = StringTransport()
        self.server = imap4.IMAP4Server()
        self.server.state = 'select'
        self.server.makeConnection(self.transport)

    def test_fetchWithPartialValidArgument(self):
        """
        If by any chance, extra bytes got appended at the end of a valid
        FETCH arguments, the client should get a BAD - arguments invalid
        response.

        See U{RFC 3501<http://tools.ietf.org/html/rfc3501#section-6.4.5>},
        section 6.4.5,
        """
        self.transport.clear()
        self.server.dataReceived(b'0001 FETCH 1 FULLL\r\n')
        expected = b'0001 BAD Illegal syntax: Invalid Argument\r\n'
        self.assertEqual(self.transport.value(), expected)
        self.transport.clear()
        self.server.connectionLost(error.ConnectionDone('Connection closed'))
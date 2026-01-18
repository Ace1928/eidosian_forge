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
class IMAP4HelperMixin:
    serverCTX: Optional[ServerTLSContext] = None
    clientCTX: Optional[ClientTLSContext] = None

    def setUp(self):
        d = defer.Deferred()
        self.server = SimpleServer(contextFactory=self.serverCTX)
        self.client = SimpleClient(d, contextFactory=self.clientCTX)
        self.connected = d
        SimpleMailbox.messages = []
        theAccount = Account(b'testuser')
        theAccount.mboxType = SimpleMailbox
        SimpleServer.theAccount = theAccount

    def tearDown(self):
        del self.server
        del self.client
        del self.connected

    def _cbStopClient(self, ignore):
        self.client.transport.loseConnection()

    def _ebGeneral(self, failure):
        self.client.transport.loseConnection()
        self.server.transport.loseConnection()
        log.err(failure, 'Problem with ' + str(self))

    def loopback(self):
        return loopback.loopbackAsync(self.server, self.client)

    def assertClientFailureMessage(self, failure, expected):
        """
        Assert that the provided failure is an L{IMAP4Exception} with
        the given message.

        @param failure: A failure whose value L{IMAP4Exception}
        @type failure: L{failure.Failure}

        @param expected: The expected failure message.
        @type expected: L{bytes}
        """
        failure.trap(imap4.IMAP4Exception)
        message = str(failure.value)
        expected = repr(expected)
        self.assertEqual(message, expected)
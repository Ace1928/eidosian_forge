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
class PipeliningTests(TestCase):
    """
    Tests for various aspects of the IMAP4 server's pipelining support.
    """
    messages = [FakeyMessage({}, [], b'', b'0', None, None), FakeyMessage({}, [], b'', b'1', None, None), FakeyMessage({}, [], b'', b'2', None, None)]

    def setUp(self):
        self.iterators = []
        self.transport = StringTransport()
        self.server = imap4.IMAP4Server(None, None, self.iterateInReactor)
        self.server.makeConnection(self.transport)
        mailbox = SynchronousMailbox(self.messages)
        self.server.state = 'select'
        self.server.mbox = mailbox
        self.transport.clear()

    def iterateInReactor(self, iterator):
        """
        A fake L{imap4.iterateInReactor} that records the iterators it
        receives.

        @param iterator: An iterator.

        @return: A L{Deferred} associated with this iterator.
        """
        d = defer.Deferred()
        self.iterators.append((iterator, d))
        return d

    def flushPending(self, asLongAs=lambda: True):
        """
        Advance pending iterators enqueued with L{iterateInReactor} in
        a round-robin fashion, resuming the transport's producer until
        it has completed.  This ensures bodies are flushed.

        @param asLongAs: (optional) An optional predicate function.
            Flushing iterators continues as long as there are
            iterators and this returns L{True}.
        """
        while self.iterators and asLongAs():
            for e in self.iterators[0][0]:
                while self.transport.producer:
                    self.transport.producer.resumeProducing()
            else:
                self.iterators.pop(0)[1].callback(None)

    def tearDown(self):
        self.server.connectionLost(failure.Failure(error.ConnectionDone()))

    def test_synchronousFetch(self):
        """
        Test that pipelined FETCH commands which can be responded to
        synchronously are responded to correctly.
        """
        self.server.dataReceived(b'01 FETCH 1 BODY[]\r\n02 FETCH 2 BODY[]\r\n03 FETCH 3 BODY[]\r\n')
        self.flushPending()
        self.assertEqual(self.transport.value(), b''.join([b'* 1 FETCH (BODY[] )\r\n', networkString('01 OK FETCH completed\r\n{5}\r\n\r\n\r\n%s' % (nativeString(self.messages[0].getBodyFile().read()),)), b'* 2 FETCH (BODY[] )\r\n', networkString('02 OK FETCH completed\r\n{5}\r\n\r\n\r\n%s' % (nativeString(self.messages[1].getBodyFile().read()),)), b'* 3 FETCH (BODY[] )\r\n', networkString('03 OK FETCH completed\r\n{5}\r\n\r\n\r\n%s' % (nativeString(self.messages[2].getBodyFile().read()),))]))

    def test_bufferedServerStatus(self):
        """
        When a server status change occurs during an ongoing FETCH
        command, the server status is buffered until the FETCH
        completes.
        """
        self.server.dataReceived(b'01 FETCH 1,2 BODY[]\r\n')
        twice = functools.partial(next, iter([True, True, False]))
        self.flushPending(asLongAs=twice)
        self.assertEqual(self.transport.value(), b''.join([b'* 1 FETCH (BODY[] )\r\n', networkString('{5}\r\n\r\n\r\n%s' % (nativeString(self.messages[0].getBodyFile().read()),))]))
        self.transport.clear()
        self.server.modeChanged(writeable=True)
        self.assertFalse(self.transport.value())
        self.flushPending()
        self.assertEqual(self.transport.value(), b''.join([b'* 2 FETCH (BODY[] )\r\n', b'* [READ-WRITE]\r\n', networkString('01 OK FETCH completed\r\n{5}\r\n\r\n\r\n%s' % (nativeString(self.messages[1].getBodyFile().read()),))]))
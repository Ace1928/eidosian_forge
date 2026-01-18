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
class LiteralTestsMixin:
    """
    Shared tests for literal classes.

    @ivar literalFactory: A callable that returns instances of the
        literal under test.
    """

    def setUp(self):
        """
        Shared setup.
        """
        self.deferred = defer.Deferred()

    def test_partialWrite(self):
        """
        The literal returns L{None} when given less data than the
        literal requires.
        """
        literal = self.literalFactory(1024, self.deferred)
        self.assertIs(None, literal.write(b'incomplete'))
        self.assertNoResult(self.deferred)

    def test_exactWrite(self):
        """
        The literal returns an empty L{bytes} instance when given
        exactly the data the literal requires.
        """
        data = b'complete'
        literal = self.literalFactory(len(data), self.deferred)
        leftover = literal.write(data)
        self.assertIsInstance(leftover, bytes)
        self.assertFalse(leftover)
        self.assertNoResult(self.deferred)

    def test_overlongWrite(self):
        """
        The literal returns any left over L{bytes} when given more
        data than the literal requires.
        """
        data = b'completeleftover'
        literal = self.literalFactory(len(b'complete'), self.deferred)
        leftover = literal.write(data)
        self.assertEqual(leftover, b'leftover')

    def test_emptyLiteral(self):
        """
        The literal returns an empty L{bytes} instance
        when given an empty L{bytes} instance.
        """
        literal = self.literalFactory(0, self.deferred)
        data = b'leftover'
        leftover = literal.write(data)
        self.assertEqual(leftover, data)
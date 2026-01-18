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
class SASLPLAINTests(TestCase):
    """
    Tests for I{SASL PLAIN} authentication, as implemented by
    L{imap4.PLAINAuthenticator} and L{imap4.PLAINCredentials}.

    @see: U{http://www.faqs.org/rfcs/rfc2595.html}
    @see: U{http://www.faqs.org/rfcs/rfc4616.html}
    """

    def test_authenticatorChallengeResponse(self):
        """
        L{PLAINAuthenticator.challengeResponse} returns challenge strings of
        the form::

            NUL<authn-id>NUL<secret>
        """
        username = b'testuser'
        secret = b'secret'
        chal = b'challenge'
        cAuth = imap4.PLAINAuthenticator(username)
        response = cAuth.challengeResponse(secret, chal)
        self.assertEqual(response, b'\x00' + username + b'\x00' + secret)

    def test_credentialsSetResponse(self):
        """
        L{PLAINCredentials.setResponse} parses challenge strings of the
        form::

            NUL<authn-id>NUL<secret>
        """
        cred = imap4.PLAINCredentials()
        cred.setResponse(b'\x00testuser\x00secret')
        self.assertEqual(cred.username, b'testuser')
        self.assertEqual(cred.password, b'secret')

    def test_credentialsInvalidResponse(self):
        """
        L{PLAINCredentials.setResponse} raises L{imap4.IllegalClientResponse}
        when passed a string not of the expected form.
        """
        cred = imap4.PLAINCredentials()
        self.assertRaises(imap4.IllegalClientResponse, cred.setResponse, b'hello')
        self.assertRaises(imap4.IllegalClientResponse, cred.setResponse, b'hello\x00world')
        self.assertRaises(imap4.IllegalClientResponse, cred.setResponse, b'hello\x00world\x00Zoom!\x00')
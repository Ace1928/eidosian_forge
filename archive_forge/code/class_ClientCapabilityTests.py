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
class ClientCapabilityTests(TestCase):
    """
    Tests for issuance of the CAPABILITY command and handling of its response.
    """

    def setUp(self):
        """
        Create an L{imap4.IMAP4Client} connected to a L{StringTransport}.
        """
        self.transport = StringTransport()
        self.protocol = imap4.IMAP4Client()
        self.protocol.makeConnection(self.transport)
        self.protocol.dataReceived(b'* OK [IMAP4rev1]\r\n')

    def test_simpleAtoms(self):
        """
        A capability response consisting only of atoms without C{'='} in them
        should result in a dict mapping those atoms to L{None}.
        """
        capabilitiesResult = self.protocol.getCapabilities(useCache=False)
        self.protocol.dataReceived(b'* CAPABILITY IMAP4rev1 LOGINDISABLED\r\n')
        self.protocol.dataReceived(b'0001 OK Capability completed.\r\n')

        def gotCapabilities(capabilities):
            self.assertEqual(capabilities, {b'IMAP4rev1': None, b'LOGINDISABLED': None})
        capabilitiesResult.addCallback(gotCapabilities)
        return capabilitiesResult

    def test_categoryAtoms(self):
        """
        A capability response consisting of atoms including C{'='} should have
        those atoms split on that byte and have capabilities in the same
        category aggregated into lists in the resulting dictionary.

        (n.b. - I made up the word "category atom"; the protocol has no notion
        of structure here, but rather allows each capability to define the
        semantics of its entry in the capability response in a freeform manner.
        If I had realized this earlier, the API for capabilities would look
        different.  As it is, we can hope that no one defines any crazy
        semantics which are incompatible with this API, or try to figure out a
        better API when someone does. -exarkun)
        """
        capabilitiesResult = self.protocol.getCapabilities(useCache=False)
        self.protocol.dataReceived(b'* CAPABILITY IMAP4rev1 AUTH=LOGIN AUTH=PLAIN\r\n')
        self.protocol.dataReceived(b'0001 OK Capability completed.\r\n')

        def gotCapabilities(capabilities):
            self.assertEqual(capabilities, {b'IMAP4rev1': None, b'AUTH': [b'LOGIN', b'PLAIN']})
        capabilitiesResult.addCallback(gotCapabilities)
        return capabilitiesResult

    def test_mixedAtoms(self):
        """
        A capability response consisting of both simple and category atoms of
        the same type should result in a list containing L{None} as well as the
        values for the category.
        """
        capabilitiesResult = self.protocol.getCapabilities(useCache=False)
        self.protocol.dataReceived(b'* CAPABILITY IMAP4rev1 FOO FOO=BAR BAR=FOO BAR\r\n')
        self.protocol.dataReceived(b'0001 OK Capability completed.\r\n')

        def gotCapabilities(capabilities):
            self.assertEqual(capabilities, {b'IMAP4rev1': None, b'FOO': [None, b'BAR'], b'BAR': [b'FOO', None]})
        capabilitiesResult.addCallback(gotCapabilities)
        return capabilitiesResult
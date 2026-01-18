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
def _fetchWork(self, fetch):

    def result(R):
        self.result = R
    self.connected.addCallback(strip(fetch)).addCallback(result).addCallback(self._cbStopClient).addErrback(self._ebGeneral)

    def check(ignored):
        self.assertFalse(self.result is self.expected)
        self.parts and self.parts.sort()
        self.server_received_parts and self.server_received_parts.sort()
        if self.uid:
            for k, v in self.expected.items():
                v['UID'] = str(k)
        self.assertEqual(self.result, self.expected)
        self.assertEqual(self.uid, self.server_received_uid)
        self.assertEqual(self.parts, self.server_received_parts)
        self.assertEqual(imap4.parseIdList(self.messages), imap4.parseIdList(self.server_received_messages))
    d = loopback.loopbackTCP(self.server, self.client, noisy=False)
    d.addCallback(check)
    return d
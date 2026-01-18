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
class PreauthIMAP4ClientMixin:
    """
    Mixin for L{SynchronousTestCase} subclasses which
    provides a C{setUp} method which creates an L{IMAP4Client}
    connected to a L{StringTransport} and puts it into the
    I{authenticated} state.

    @ivar transport: A L{StringTransport} to which C{client} is
        connected.

    @ivar client: An L{IMAP4Client} which is connected to
        C{transport}.
    """
    clientProtocol: Type[imap4.IMAP4Client] = imap4.IMAP4Client

    def setUp(self):
        """
        Create an IMAP4Client connected to a fake transport and in the
        authenticated state.
        """
        self.transport = StringTransport()
        self.client = self.clientProtocol()
        self.client.makeConnection(self.transport)
        self.client.dataReceived(b'* PREAUTH Hello unittest\r\n')
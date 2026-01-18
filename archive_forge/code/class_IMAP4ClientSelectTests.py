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
class IMAP4ClientSelectTests(SelectionTestsMixin, SynchronousTestCase):
    """
    Tests for the L{IMAP4Client.select} method.

    An example of usage of the SELECT command from RFC 3501, section 6.3.1::

        C: A142 SELECT INBOX
        S: * 172 EXISTS
        S: * 1 RECENT
        S: * OK [UNSEEN 12] Message 12 is first unseen
        S: * OK [UIDVALIDITY 3857529045] UIDs valid
        S: * OK [UIDNEXT 4392] Predicted next UID
        S: * FLAGS (\\Answered \\Flagged \\Deleted \\Seen \\Draft)
        S: * OK [PERMANENTFLAGS (\\Deleted \\Seen \\*)] Limited
        S: A142 OK [READ-WRITE] SELECT completed
    """
    method = 'select'
    command = b'SELECT'
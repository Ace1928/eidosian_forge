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
@implementer(imap4.IMailboxInfo, imap4.IMailbox, imap4.ICloseableMailbox)
class SimpleMailbox:
    flags = ('\\Flag1', 'Flag2', '\\AnotherSysFlag', 'LastFlag')
    messages: list[tuple[bytes, list[bytes], bytes, int]] = []
    mUID = 0
    rw = 1
    closed = False

    def __init__(self):
        self.listeners = []
        self.addListener = self.listeners.append
        self.removeListener = self.listeners.remove

    def getFlags(self):
        return self.flags

    def getUIDValidity(self):
        return 42

    def getUIDNext(self):
        return len(self.messages) + 1

    def getMessageCount(self):
        return 9

    def getRecentCount(self):
        return 3

    def getUnseenCount(self):
        return 4

    def isWriteable(self):
        return self.rw

    def destroy(self):
        pass

    def getHierarchicalDelimiter(self):
        return '/'

    def requestStatus(self, names):
        r = {}
        if 'MESSAGES' in names:
            r['MESSAGES'] = self.getMessageCount()
        if 'RECENT' in names:
            r['RECENT'] = self.getRecentCount()
        if 'UIDNEXT' in names:
            r['UIDNEXT'] = self.getMessageCount() + 1
        if 'UIDVALIDITY' in names:
            r['UIDVALIDITY'] = self.getUID()
        if 'UNSEEN' in names:
            r['UNSEEN'] = self.getUnseenCount()
        return defer.succeed(r)

    def addMessage(self, message, flags, date=None):
        self.messages.append((message, flags, date, self.mUID))
        self.mUID += 1
        return defer.succeed(None)

    def expunge(self):
        delete = []
        for i in self.messages:
            if '\\Deleted' in i[1]:
                delete.append(i)
        for i in delete:
            self.messages.remove(i)
        return [i[3] for i in delete]

    def close(self):
        self.closed = True

    def fetch(self, messages, uid):
        pass

    def getUID(self, message):
        pass

    def store(self, messages, flags, mode, uid):
        pass
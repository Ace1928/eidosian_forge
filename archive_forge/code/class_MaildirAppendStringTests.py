import email.message
import email.parser
import errno
import glob
import io
import os
import pickle
import shutil
import signal
import sys
import tempfile
import textwrap
import time
from hashlib import md5
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.portal
import twisted.mail.alias
import twisted.mail.mail
import twisted.mail.maildir
import twisted.mail.protocols
import twisted.mail.relay
import twisted.mail.relaymanager
from twisted import cred, mail
from twisted.internet import address, defer, interfaces, protocol, reactor, task
from twisted.internet.defer import Deferred
from twisted.internet.error import (
from twisted.internet.testing import (
from twisted.mail import pop3, smtp
from twisted.mail.relaymanager import _AttemptManager
from twisted.names import dns
from twisted.names.dns import Record_CNAME, Record_MX, RRHeader
from twisted.names.error import DNSNameError
from twisted.python import failure, log
from twisted.python.filepath import FilePath
from twisted.python.runtime import platformType
from twisted.trial.unittest import TestCase
from twisted.names import client, common, server
@skipIf(platformType != 'posix', 'twisted.mail only works on posix')
class MaildirAppendStringTests(TestCase, _AppendTestMixin):
    """
    Tests for L{MaildirMailbox.appendMessage} when invoked with a C{str}.
    """

    def setUp(self):
        self.d = self.mktemp()
        mail.maildir.initializeMaildir(self.d)

    def _append(self, ignored, mbox):
        d = mbox.appendMessage('TEST')
        return self.assertFailure(d, Exception)

    def _setState(self, ignored, mbox, rename=None, write=None, open=None):
        """
        Change the behavior of future C{rename}, C{write}, or C{open} calls made
        by the mailbox C{mbox}.

        @param rename: If not L{None}, a new value for the C{_renamestate}
            attribute of the mailbox's append factory.  The original value will
            be restored at the end of the test.

        @param write: Like C{rename}, but for the C{_writestate} attribute.

        @param open: Like C{rename}, but for the C{_openstate} attribute.
        """
        if rename is not None:
            self.addCleanup(setattr, mbox.AppendFactory, '_renamestate', mbox.AppendFactory._renamestate)
            mbox.AppendFactory._renamestate = rename
        if write is not None:
            self.addCleanup(setattr, mbox.AppendFactory, '_writestate', mbox.AppendFactory._writestate)
            mbox.AppendFactory._writestate = write
        if open is not None:
            self.addCleanup(setattr, mbox.AppendFactory, '_openstate', mbox.AppendFactory._openstate)
            mbox.AppendFactory._openstate = open

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def test_append(self):
        """
        L{MaildirMailbox.appendMessage} returns a L{Deferred} which fires when
        the message has been added to the end of the mailbox.
        """
        mbox = mail.maildir.MaildirMailbox(self.d)
        mbox.AppendFactory = FailingMaildirMailboxAppendMessageTask
        d = self._appendMessages(mbox, ['X' * i for i in range(1, 11)])
        d.addCallback(self.assertEqual, [None] * 10)
        d.addCallback(self._cbTestAppend, mbox)
        return d

    def _cbTestAppend(self, ignored, mbox):
        """
        Check that the mailbox has the expected number (ten) of messages in it,
        and that each has the expected contents, and that they are in the same
        order as that in which they were appended.
        """
        self.assertEqual(len(mbox.listMessages()), 10)
        self.assertEqual([len(mbox.getMessage(i).read()) for i in range(10)], list(range(1, 11)))
        self._setState(None, mbox, rename=False)
        d = self._append(None, mbox)
        d.addCallback(self._setState, mbox, rename=True, write=False)
        d.addCallback(self._append, mbox)
        d.addCallback(self._setState, mbox, write=True, open=False)
        d.addCallback(self._append, mbox)
        d.addCallback(self._setState, mbox, open=True)
        return d
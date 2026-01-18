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
class MaildirAppendFileTests(TestCase, _AppendTestMixin):
    """
    Tests for L{MaildirMailbox.appendMessage} when invoked with a C{str}.
    """

    def setUp(self):
        self.d = self.mktemp()
        mail.maildir.initializeMaildir(self.d)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def test_append(self):
        """
        L{MaildirMailbox.appendMessage} returns a L{Deferred} which fires when
        the message has been added to the end of the mailbox.
        """
        mbox = mail.maildir.MaildirMailbox(self.d)
        messages = []
        for i in range(1, 11):
            temp = tempfile.TemporaryFile()
            temp.write('X' * i)
            temp.seek(0, 0)
            messages.append(temp)
            self.addCleanup(temp.close)
        d = self._appendMessages(mbox, messages)
        d.addCallback(self._cbTestAppend, mbox)
        return d

    def _cbTestAppend(self, result, mbox):
        """
        Check that the mailbox has the expected number (ten) of messages in it,
        and that each has the expected contents, and that they are in the same
        order as that in which they were appended.
        """
        self.assertEqual(len(mbox.listMessages()), 10)
        self.assertEqual([len(mbox.getMessage(i).read()) for i in range(10)], list(range(1, 11)))
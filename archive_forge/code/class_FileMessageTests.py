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
class FileMessageTests(TestCase):

    def setUp(self):
        self.name = self.mktemp()
        self.final = self.mktemp()
        self.f = open(self.name, 'wb')
        self.addCleanup(self.f.close)
        self.fp = mail.mail.FileMessage(self.f, self.name, self.final)

    def testFinalName(self):
        return self.fp.eomReceived().addCallback(self._cbFinalName)

    def _cbFinalName(self, result):
        self.assertEqual(result, self.final)
        self.assertTrue(self.f.closed)
        self.assertFalse(os.path.exists(self.name))

    def testContents(self):
        contents = b'first line\nsecond line\nthird line\n'
        for line in contents.splitlines():
            self.fp.lineReceived(line)
        self.fp.eomReceived()
        with open(self.final, 'rb') as f:
            self.assertEqual(f.read(), contents)

    def testInterrupted(self):
        contents = b'first line\nsecond line\n'
        for line in contents.splitlines():
            self.fp.lineReceived(line)
        self.fp.connectionLost()
        self.assertFalse(os.path.exists(self.name))
        self.assertFalse(os.path.exists(self.final))
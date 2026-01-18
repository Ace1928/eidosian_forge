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
class MaildirDirdbmDomainTests(TestCase):
    """
    Tests for L{MaildirDirdbmDomain}.
    """

    def setUp(self):
        """
        Create a temporary L{MaildirDirdbmDomain} and parent
        L{MailService} before running each test.
        """
        self.P = self.mktemp()
        self.S = mail.mail.MailService()
        self.D = mail.maildir.MaildirDirdbmDomain(self.S, self.P)

    def tearDown(self):
        """
        Remove the temporary C{maildir} directory when the test has
        finished.
        """
        shutil.rmtree(self.P)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def test_addUser(self):
        """
        L{MaildirDirdbmDomain.addUser} accepts a user and password
        argument. It stores those in a C{dbm} dictionary
        attribute and creates a directory for each user.
        """
        toAdd = (('user1', 'pwd1'), ('user2', 'pwd2'), ('user3', 'pwd3'))
        for u, p in toAdd:
            self.D.addUser(u, p)
        for u, p in toAdd:
            self.assertTrue(u in self.D.dbm)
            self.assertEqual(self.D.dbm[u], p)
            self.assertTrue(os.path.exists(os.path.join(self.P, u)))

    def test_credentials(self):
        """
        L{MaildirDirdbmDomain.getCredentialsCheckers} initializes and
        returns one L{ICredentialsChecker} checker by default.
        """
        creds = self.D.getCredentialsCheckers()
        self.assertEqual(len(creds), 1)
        self.assertTrue(cred.checkers.ICredentialsChecker.providedBy(creds[0]))
        self.assertTrue(cred.credentials.IUsernamePassword in creds[0].credentialInterfaces)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def test_requestAvatar(self):
        """
        L{MaildirDirdbmDomain.requestAvatar} raises L{NotImplementedError}
        unless it is supplied with an L{pop3.IMailbox} interface.
        When called with an L{pop3.IMailbox}, it returns a 3-tuple
        containing L{pop3.IMailbox}, an implementation of that interface
        and a NOOP callable.
        """

        class ISomething(Interface):
            pass
        self.D.addUser('user', 'password')
        self.assertRaises(NotImplementedError, self.D.requestAvatar, 'user', None, ISomething)
        t = self.D.requestAvatar('user', None, pop3.IMailbox)
        self.assertEqual(len(t), 3)
        self.assertTrue(t[0] is pop3.IMailbox)
        self.assertTrue(pop3.IMailbox.providedBy(t[1]))
        t[2]()

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def test_requestAvatarId(self):
        """
        L{DirdbmDatabase.requestAvatarId} raises L{UnauthorizedLogin} if
        supplied with invalid user credentials.
        When called with valid credentials, L{requestAvatarId} returns
        the username associated with the supplied credentials.
        """
        self.D.addUser('user', 'password')
        database = self.D.getCredentialsCheckers()[0]
        creds = cred.credentials.UsernamePassword('user', 'wrong password')
        self.assertRaises(cred.error.UnauthorizedLogin, database.requestAvatarId, creds)
        creds = cred.credentials.UsernamePassword('user', 'password')
        self.assertEqual(database.requestAvatarId(creds), 'user')

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def test_userDirectory(self):
        """
        L{MaildirDirdbmDomain.userDirectory} is supplied with a user name
        and returns the path to that user's maildir subdirectory.
        Calling L{MaildirDirdbmDomain.userDirectory} with a
        non-existent user returns the 'postmaster' directory if there
        is a postmaster or returns L{None} if there is no postmaster.
        """
        self.D.addUser('user', 'password')
        self.assertEqual(self.D.userDirectory('user'), os.path.join(self.D.root, 'user'))
        self.D.postmaster = False
        self.assertIdentical(self.D.userDirectory('nouser'), None)
        self.D.postmaster = True
        self.assertEqual(self.D.userDirectory('nouser'), os.path.join(self.D.root, 'postmaster'))
import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class FTPRealmTests(TestCase):
    """
    Tests for L{ftp.FTPRealm}.
    """

    def test_getHomeDirectory(self):
        """
        L{ftp.FTPRealm} accepts an extra directory to its initializer and treats
        the avatarId passed to L{ftp.FTPRealm.getHomeDirectory} as a single path
        segment to construct a child of that directory.
        """
        base = '/path/to/home'
        realm = ftp.FTPRealm(self.mktemp(), base)
        home = realm.getHomeDirectory('alice@example.com')
        self.assertEqual(filepath.FilePath(base).child('alice@example.com'), home)

    def test_defaultHomeDirectory(self):
        """
        If no extra directory is passed to L{ftp.FTPRealm}, it uses C{"/home"}
        as the base directory containing all user home directories.
        """
        realm = ftp.FTPRealm(self.mktemp())
        home = realm.getHomeDirectory('alice@example.com')
        self.assertEqual(filepath.FilePath('/home/alice@example.com'), home)
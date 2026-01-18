import os
from io import StringIO
from typing import Sequence, Type
from unittest import skipIf
from zope.interface import Interface
from twisted import plugin
from twisted.cred import checkers, credentials, error, strcred
from twisted.plugins import cred_anonymous, cred_file, cred_unix
from twisted.python import usage
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
@skipIf(not pwd, 'Required module not available: pwd')
@skipIf(not crypt, 'Required module not available: crypt')
@skipIf(not spwd, 'Required module not available: spwd')
class UnixCheckerTests(TestCase):
    users = {'admin': 'asdf', 'alice': 'foo'}

    def _spwd_getspnam(self, username):
        return spwd.struct_spwd((username, crypt.crypt(self.users[username], 'F/'), 0, 0, 99999, 7, -1, -1, -1))

    def setUp(self):
        self.admin = credentials.UsernamePassword('admin', 'asdf')
        self.alice = credentials.UsernamePassword('alice', 'foo')
        self.badPass = credentials.UsernamePassword('alice', 'foobar')
        self.badUser = credentials.UsernamePassword('x', 'yz')
        self.checker = strcred.makeChecker('unix')
        self.adminBytes = credentials.UsernamePassword(b'admin', b'asdf')
        self.aliceBytes = credentials.UsernamePassword(b'alice', b'foo')
        self.badPassBytes = credentials.UsernamePassword(b'alice', b'foobar')
        self.badUserBytes = credentials.UsernamePassword(b'x', b'yz')
        self.checkerBytes = strcred.makeChecker('unix')
        if pwd:
            database = UserDatabase()
            for username, password in self.users.items():
                database.addUser(username, crypt.crypt(password, 'F/'), 1000, 1000, username, '/home/' + username, '/bin/sh')
            self.patch(pwd, 'getpwnam', database.getpwnam)
        if spwd:
            self.patch(spwd, 'getspnam', self._spwd_getspnam)

    def test_isChecker(self):
        """
        Verifies that strcred.makeChecker('unix') returns an object
        that implements the L{ICredentialsChecker} interface.
        """
        self.assertTrue(checkers.ICredentialsChecker.providedBy(self.checker))
        self.assertIn(credentials.IUsernamePassword, self.checker.credentialInterfaces)
        self.assertTrue(checkers.ICredentialsChecker.providedBy(self.checkerBytes))
        self.assertIn(credentials.IUsernamePassword, self.checkerBytes.credentialInterfaces)

    def test_unixCheckerSucceeds(self):
        """
        The checker works with valid credentials.
        """

        def _gotAvatar(username):
            self.assertEqual(username, self.admin.username)
        return self.checker.requestAvatarId(self.admin).addCallback(_gotAvatar)

    def test_unixCheckerSucceedsBytes(self):
        """
        The checker works with valid L{bytes} credentials.
        """

        def _gotAvatar(username):
            self.assertEqual(username, self.adminBytes.username.decode('utf-8'))
        return self.checkerBytes.requestAvatarId(self.adminBytes).addCallback(_gotAvatar)

    def test_unixCheckerFailsUsername(self):
        """
        The checker fails with an invalid username.
        """
        return self.assertFailure(self.checker.requestAvatarId(self.badUser), error.UnauthorizedLogin)

    def test_unixCheckerFailsUsernameBytes(self):
        """
        The checker fails with an invalid L{bytes} username.
        """
        return self.assertFailure(self.checkerBytes.requestAvatarId(self.badUserBytes), error.UnauthorizedLogin)

    def test_unixCheckerFailsPassword(self):
        """
        The checker fails with an invalid password.
        """
        return self.assertFailure(self.checker.requestAvatarId(self.badPass), error.UnauthorizedLogin)

    def test_unixCheckerFailsPasswordBytes(self):
        """
        The checker fails with an invalid L{bytes} password.
        """
        return self.assertFailure(self.checkerBytes.requestAvatarId(self.badPassBytes), error.UnauthorizedLogin)
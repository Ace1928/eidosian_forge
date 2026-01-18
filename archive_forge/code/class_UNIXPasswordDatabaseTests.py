import os
from base64 import encodebytes
from collections import namedtuple
from io import BytesIO
from typing import Optional
from zope.interface.verify import verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet.defer import Deferred
from twisted.python import util
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
class UNIXPasswordDatabaseTests(TestCase):
    """
    Tests for L{UNIXPasswordDatabase}.
    """
    skip = cryptSkip or dependencySkip

    def assertLoggedIn(self, d: Deferred[bytes], username: bytes) -> None:
        """
        Assert that the L{Deferred} passed in is called back with the value
        'username'.  This represents a valid login for this TestCase.

        @param d: a L{Deferred} from an L{IChecker.requestAvatarId} method.
        """
        self.assertEqual(self.successResultOf(d), username)

    def test_defaultCheckers(self):
        """
        L{UNIXPasswordDatabase} with no arguments has checks the C{pwd} database
        and then the C{spwd} database.
        """
        checker = checkers.UNIXPasswordDatabase()

        def crypted(username, password):
            salt = crypt.crypt(password, username)
            crypted = crypt.crypt(password, '$1$' + salt)
            return crypted
        pwd = UserDatabase()
        pwd.addUser('alice', crypted('alice', 'password'), 1, 2, 'foo', '/foo', '/bin/sh')
        pwd.addUser('bob', 'x', 1, 2, 'bar', '/bar', '/bin/sh')
        spwd = ShadowDatabase()
        spwd.addUser('alice', 'wrong', 1, 2, 3, 4, 5, 6, 7)
        spwd.addUser('bob', crypted('bob', 'password'), 8, 9, 10, 11, 12, 13, 14)
        self.patch(checkers, 'pwd', pwd)
        self.patch(checkers, 'spwd', spwd)
        mockos = MockOS()
        self.patch(util, 'os', mockos)
        mockos.euid = 2345
        mockos.egid = 1234
        cred = UsernamePassword(b'alice', b'password')
        self.assertLoggedIn(checker.requestAvatarId(cred), b'alice')
        self.assertEqual(mockos.seteuidCalls, [])
        self.assertEqual(mockos.setegidCalls, [])
        cred.username = b'bob'
        self.assertLoggedIn(checker.requestAvatarId(cred), b'bob')
        self.assertEqual(mockos.seteuidCalls, [0, 2345])
        self.assertEqual(mockos.setegidCalls, [0, 1234])

    def assertUnauthorizedLogin(self, d):
        """
        Asserts that the L{Deferred} passed in is erred back with an
        L{UnauthorizedLogin} L{Failure}.  This reprsents an invalid login for
        this TestCase.

        NOTE: To work, this method's return value must be returned from the
        test method, or otherwise hooked up to the test machinery.

        @param d: a L{Deferred} from an L{IChecker.requestAvatarId} method.
        @type d: L{Deferred}
        @rtype: L{None}
        """
        self.failureResultOf(d, checkers.UnauthorizedLogin)

    def test_passInCheckers(self):
        """
        L{UNIXPasswordDatabase} takes a list of functions to check for UNIX
        user information.
        """
        password = crypt.crypt('secret', 'secret')
        userdb = UserDatabase()
        userdb.addUser('anybody', password, 1, 2, 'foo', '/bar', '/bin/sh')
        checker = checkers.UNIXPasswordDatabase([userdb.getpwnam])
        self.assertLoggedIn(checker.requestAvatarId(UsernamePassword(b'anybody', b'secret')), b'anybody')

    def test_verifyPassword(self):
        """
        If the encrypted password provided by the getpwnam function is valid
        (verified by the L{verifyCryptedPassword} function), we callback the
        C{requestAvatarId} L{Deferred} with the username.
        """

        def verifyCryptedPassword(crypted, pw):
            return crypted == pw

        def getpwnam(username):
            return [username, username]
        self.patch(checkers, 'verifyCryptedPassword', verifyCryptedPassword)
        checker = checkers.UNIXPasswordDatabase([getpwnam])
        credential = UsernamePassword(b'username', b'username')
        self.assertLoggedIn(checker.requestAvatarId(credential), b'username')

    def test_failOnKeyError(self):
        """
        If the getpwnam function raises a KeyError, the login fails with an
        L{UnauthorizedLogin} exception.
        """

        def getpwnam(username):
            raise KeyError(username)
        checker = checkers.UNIXPasswordDatabase([getpwnam])
        credential = UsernamePassword(b'username', b'password')
        self.assertUnauthorizedLogin(checker.requestAvatarId(credential))

    def test_failOnBadPassword(self):
        """
        If the verifyCryptedPassword function doesn't verify the password, the
        login fails with an L{UnauthorizedLogin} exception.
        """

        def verifyCryptedPassword(crypted, pw):
            return False

        def getpwnam(username):
            return [username, b'password']
        self.patch(checkers, 'verifyCryptedPassword', verifyCryptedPassword)
        checker = checkers.UNIXPasswordDatabase([getpwnam])
        credential = UsernamePassword(b'username', b'password')
        self.assertUnauthorizedLogin(checker.requestAvatarId(credential))

    def test_loopThroughFunctions(self):
        """
        UNIXPasswordDatabase.requestAvatarId loops through each getpwnam
        function associated with it and returns a L{Deferred} which fires with
        the result of the first one which returns a value other than None.
        ones do not verify the password.
        """

        def verifyCryptedPassword(crypted, pw):
            return crypted == pw

        def getpwnam1(username):
            return [username, 'not the password']

        def getpwnam2(username):
            return [username, 'password']
        self.patch(checkers, 'verifyCryptedPassword', verifyCryptedPassword)
        checker = checkers.UNIXPasswordDatabase([getpwnam1, getpwnam2])
        credential = UsernamePassword(b'username', b'password')
        self.assertLoggedIn(checker.requestAvatarId(credential), b'username')

    def test_failOnSpecial(self):
        """
        If the password returned by any function is C{""}, C{"x"}, or C{"*"} it
        is not compared against the supplied password.  Instead it is skipped.
        """
        pwd = UserDatabase()
        pwd.addUser('alice', '', 1, 2, '', 'foo', 'bar')
        pwd.addUser('bob', 'x', 1, 2, '', 'foo', 'bar')
        pwd.addUser('carol', '*', 1, 2, '', 'foo', 'bar')
        self.patch(checkers, 'pwd', pwd)
        checker = checkers.UNIXPasswordDatabase([checkers._pwdGetByName])
        cred = UsernamePassword(b'alice', b'')
        self.assertUnauthorizedLogin(checker.requestAvatarId(cred))
        cred = UsernamePassword(b'bob', b'x')
        self.assertUnauthorizedLogin(checker.requestAvatarId(cred))
        cred = UsernamePassword(b'carol', b'*')
        self.assertUnauthorizedLogin(checker.requestAvatarId(cred))
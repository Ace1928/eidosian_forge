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
class SSHPublicKeyDatabaseTests(TestCase):
    """
    Tests for L{SSHPublicKeyDatabase}.
    """
    skip = euidSkip or dependencySkip

    def setUp(self) -> None:
        self.checker = checkers.SSHPublicKeyDatabase()
        self.key1 = encodebytes(b'foobar')
        self.key2 = encodebytes(b'eggspam')
        self.content = b't1 ' + self.key1 + b' foo\nt2 ' + self.key2 + b' egg\n'
        self.mockos = MockOS()
        self.patch(util, 'os', self.mockos)
        self.path = FilePath(self.mktemp())
        assert isinstance(self.path.path, str)
        self.sshDir = self.path.child('.ssh')
        self.sshDir.makedirs()
        userdb = UserDatabase()
        userdb.addUser('user', 'password', 1, 2, 'first last', self.path.path, '/bin/shell')
        self.checker._userdb = userdb

    def test_deprecated(self):
        """
        L{SSHPublicKeyDatabase} is deprecated as of version 15.0
        """
        warningsShown = self.flushWarnings(offendingFunctions=[self.setUp])
        self.assertEqual(warningsShown[0]['category'], DeprecationWarning)
        self.assertEqual(warningsShown[0]['message'], 'twisted.conch.checkers.SSHPublicKeyDatabase was deprecated in Twisted 15.0.0: Please use twisted.conch.checkers.SSHPublicKeyChecker, initialized with an instance of twisted.conch.checkers.UNIXAuthorizedKeysFiles instead.')
        self.assertEqual(len(warningsShown), 1)

    def _testCheckKey(self, filename):
        self.sshDir.child(filename).setContent(self.content)
        user = UsernamePassword(b'user', b'password')
        user.blob = b'foobar'
        self.assertTrue(self.checker.checkKey(user))
        user.blob = b'eggspam'
        self.assertTrue(self.checker.checkKey(user))
        user.blob = b'notallowed'
        self.assertFalse(self.checker.checkKey(user))

    def test_checkKey(self):
        """
        L{SSHPublicKeyDatabase.checkKey} should retrieve the content of the
        authorized_keys file and check the keys against that file.
        """
        self._testCheckKey('authorized_keys')
        self.assertEqual(self.mockos.seteuidCalls, [])
        self.assertEqual(self.mockos.setegidCalls, [])

    def test_checkKey2(self):
        """
        L{SSHPublicKeyDatabase.checkKey} should retrieve the content of the
        authorized_keys2 file and check the keys against that file.
        """
        self._testCheckKey('authorized_keys2')
        self.assertEqual(self.mockos.seteuidCalls, [])
        self.assertEqual(self.mockos.setegidCalls, [])

    def test_checkKeyAsRoot(self):
        """
        If the key file is readable, L{SSHPublicKeyDatabase.checkKey} should
        switch its uid/gid to the ones of the authenticated user.
        """
        keyFile = self.sshDir.child('authorized_keys')
        keyFile.setContent(self.content)
        keyFile.chmod(0)
        self.addCleanup(keyFile.chmod, 511)
        savedSeteuid = self.mockos.seteuid

        def seteuid(euid):
            keyFile.chmod(511)
            return savedSeteuid(euid)
        self.mockos.euid = 2345
        self.mockos.egid = 1234
        self.patch(self.mockos, 'seteuid', seteuid)
        self.patch(util, 'os', self.mockos)
        user = UsernamePassword(b'user', b'password')
        user.blob = b'foobar'
        self.assertTrue(self.checker.checkKey(user))
        self.assertEqual(self.mockos.seteuidCalls, [0, 1, 0, 2345])
        self.assertEqual(self.mockos.setegidCalls, [2, 1234])

    def test_requestAvatarId(self):
        """
        L{SSHPublicKeyDatabase.requestAvatarId} should return the avatar id
        passed in if its C{_checkKey} method returns True.
        """

        def _checkKey(ignored):
            return True
        self.patch(self.checker, 'checkKey', _checkKey)
        credentials = SSHPrivateKey(b'test', b'ssh-rsa', keydata.publicRSA_openssh, b'foo', keys.Key.fromString(keydata.privateRSA_openssh).sign(b'foo'))
        d = self.checker.requestAvatarId(credentials)

        def _verify(avatarId):
            self.assertEqual(avatarId, b'test')
        return d.addCallback(_verify)

    def test_requestAvatarIdWithoutSignature(self):
        """
        L{SSHPublicKeyDatabase.requestAvatarId} should raise L{ValidPublicKey}
        if the credentials represent a valid key without a signature.  This
        tells the user that the key is valid for login, but does not actually
        allow that user to do so without a signature.
        """

        def _checkKey(ignored):
            return True
        self.patch(self.checker, 'checkKey', _checkKey)
        credentials = SSHPrivateKey(b'test', b'ssh-rsa', keydata.publicRSA_openssh, None, None)
        d = self.checker.requestAvatarId(credentials)
        return self.assertFailure(d, ValidPublicKey)

    def test_requestAvatarIdInvalidKey(self):
        """
        If L{SSHPublicKeyDatabase.checkKey} returns False,
        C{_cbRequestAvatarId} should raise L{UnauthorizedLogin}.
        """

        def _checkKey(ignored):
            return False
        self.patch(self.checker, 'checkKey', _checkKey)
        d = self.checker.requestAvatarId(None)
        return self.assertFailure(d, UnauthorizedLogin)

    def test_requestAvatarIdInvalidSignature(self):
        """
        Valid keys with invalid signatures should cause
        L{SSHPublicKeyDatabase.requestAvatarId} to return a {UnauthorizedLogin}
        failure
        """

        def _checkKey(ignored):
            return True
        self.patch(self.checker, 'checkKey', _checkKey)
        credentials = SSHPrivateKey(b'test', b'ssh-rsa', keydata.publicRSA_openssh, b'foo', keys.Key.fromString(keydata.privateDSA_openssh).sign(b'foo'))
        d = self.checker.requestAvatarId(credentials)
        return self.assertFailure(d, UnauthorizedLogin)

    def test_requestAvatarIdNormalizeException(self):
        """
        Exceptions raised while verifying the key should be normalized into an
        C{UnauthorizedLogin} failure.
        """

        def _checkKey(ignored):
            return True
        self.patch(self.checker, 'checkKey', _checkKey)
        credentials = SSHPrivateKey(b'test', None, b'blob', b'sigData', b'sig')
        d = self.checker.requestAvatarId(credentials)

        def _verifyLoggedException(failure):
            errors = self.flushLoggedErrors(keys.BadKeyError)
            self.assertEqual(len(errors), 1)
            return failure
        d.addErrback(_verifyLoggedException)
        return self.assertFailure(d, UnauthorizedLogin)
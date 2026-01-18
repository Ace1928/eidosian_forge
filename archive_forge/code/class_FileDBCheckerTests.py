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
class FileDBCheckerTests(TestCase):
    """
    C{--auth=file:...} file checker.
    """

    def setUp(self):
        self.admin = credentials.UsernamePassword(b'admin', b'asdf')
        self.alice = credentials.UsernamePassword(b'alice', b'foo')
        self.badPass = credentials.UsernamePassword(b'alice', b'foobar')
        self.badUser = credentials.UsernamePassword(b'x', b'yz')
        self.filename = self.mktemp()
        FilePath(self.filename).setContent(b'admin:asdf\nalice:foo\n')
        self.checker = strcred.makeChecker('file:' + self.filename)

    def _fakeFilename(self):
        filename = '/DoesNotExist'
        while os.path.exists(filename):
            filename += '_'
        return filename

    def test_isChecker(self):
        """
        Verifies that strcred.makeChecker('memory') returns an object
        that implements the L{ICredentialsChecker} interface.
        """
        self.assertTrue(checkers.ICredentialsChecker.providedBy(self.checker))
        self.assertIn(credentials.IUsernamePassword, self.checker.credentialInterfaces)

    def test_fileCheckerSucceeds(self):
        """
        The checker works with valid credentials.
        """

        def _gotAvatar(username):
            self.assertEqual(username, self.admin.username)
        return self.checker.requestAvatarId(self.admin).addCallback(_gotAvatar)

    def test_fileCheckerFailsUsername(self):
        """
        The checker fails with an invalid username.
        """
        return self.assertFailure(self.checker.requestAvatarId(self.badUser), error.UnauthorizedLogin)

    def test_fileCheckerFailsPassword(self):
        """
        The checker fails with an invalid password.
        """
        return self.assertFailure(self.checker.requestAvatarId(self.badPass), error.UnauthorizedLogin)

    def test_failsWithEmptyFilename(self):
        """
        An empty filename raises an error.
        """
        self.assertRaises(ValueError, strcred.makeChecker, 'file')
        self.assertRaises(ValueError, strcred.makeChecker, 'file:')

    def test_warnWithBadFilename(self):
        """
        When the file auth plugin is given a file that doesn't exist, it
        should produce a warning.
        """
        oldOutput = cred_file.theFileCheckerFactory.errorOutput
        newOutput = StringIO()
        cred_file.theFileCheckerFactory.errorOutput = newOutput
        strcred.makeChecker('file:' + self._fakeFilename())
        cred_file.theFileCheckerFactory.errorOutput = oldOutput
        self.assertIn(cred_file.invalidFileWarning, newOutput.getvalue())
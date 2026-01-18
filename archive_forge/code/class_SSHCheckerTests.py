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
@skipIf(not requireModule('cryptography'), 'cryptography is not available')
class SSHCheckerTests(TestCase):
    """
    Tests for the C{--auth=sshkey:...} checker.  The majority of the
    tests for the ssh public key database checker are in
    L{twisted.conch.test.test_checkers.SSHPublicKeyCheckerTestCase}.
    """

    def test_isChecker(self):
        """
        Verifies that strcred.makeChecker('sshkey') returns an object
        that implements the L{ICredentialsChecker} interface.
        """
        sshChecker = strcred.makeChecker('sshkey')
        self.assertTrue(checkers.ICredentialsChecker.providedBy(sshChecker))
        self.assertIn(credentials.ISSHPrivateKey, sshChecker.credentialInterfaces)
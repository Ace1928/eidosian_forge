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
class PublicAPITests(TestCase):

    def test_emptyDescription(self):
        """
        The description string cannot be empty.
        """
        iat = getInvalidAuthType()
        self.assertRaises(strcred.InvalidAuthType, strcred.makeChecker, iat)
        self.assertRaises(strcred.InvalidAuthType, strcred.findCheckerFactory, iat)

    def test_invalidAuthType(self):
        """
        An unrecognized auth type raises an exception.
        """
        iat = getInvalidAuthType()
        self.assertRaises(strcred.InvalidAuthType, strcred.makeChecker, iat)
        self.assertRaises(strcred.InvalidAuthType, strcred.findCheckerFactory, iat)
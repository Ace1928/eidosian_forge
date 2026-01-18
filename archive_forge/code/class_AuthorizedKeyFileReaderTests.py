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
class AuthorizedKeyFileReaderTests(TestCase):
    """
    Tests for L{checkers.readAuthorizedKeyFile}
    """
    skip = dependencySkip

    def test_ignoresComments(self):
        """
        L{checkers.readAuthorizedKeyFile} does not attempt to turn comments
        into keys
        """
        fileobj = BytesIO(b'# this comment is ignored\nthis is not\n# this is again\nand this is not')
        result = checkers.readAuthorizedKeyFile(fileobj, lambda x: x)
        self.assertEqual([b'this is not', b'and this is not'], list(result))

    def test_ignoresLeadingWhitespaceAndEmptyLines(self):
        """
        L{checkers.readAuthorizedKeyFile} ignores leading whitespace in
        lines, as well as empty lines
        """
        fileobj = BytesIO(b'\n                           # ignore\n                           not ignored\n                           ')
        result = checkers.readAuthorizedKeyFile(fileobj, parseKey=lambda x: x)
        self.assertEqual([b'not ignored'], list(result))

    def test_ignoresUnparsableKeys(self):
        """
        L{checkers.readAuthorizedKeyFile} does not raise an exception
        when a key fails to parse (raises a
        L{twisted.conch.ssh.keys.BadKeyError}), but rather just keeps going
        """

        def failOnSome(line):
            if line.startswith(b'f'):
                raise keys.BadKeyError('failed to parse')
            return line
        fileobj = BytesIO(b'failed key\ngood key')
        result = checkers.readAuthorizedKeyFile(fileobj, parseKey=failOnSome)
        self.assertEqual([b'good key'], list(result))
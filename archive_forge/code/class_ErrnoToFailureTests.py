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
class ErrnoToFailureTests(TestCase):
    """
    Tests for L{ftp.errnoToFailure} errno checking.
    """

    def test_notFound(self):
        """
        C{errno.ENOENT} should be translated to L{ftp.FileNotFoundError}.
        """
        d = ftp.errnoToFailure(errno.ENOENT, 'foo')
        return self.assertFailure(d, ftp.FileNotFoundError)

    def test_permissionDenied(self):
        """
        C{errno.EPERM} should be translated to L{ftp.PermissionDeniedError}.
        """
        d = ftp.errnoToFailure(errno.EPERM, 'foo')
        return self.assertFailure(d, ftp.PermissionDeniedError)

    def test_accessDenied(self):
        """
        C{errno.EACCES} should be translated to L{ftp.PermissionDeniedError}.
        """
        d = ftp.errnoToFailure(errno.EACCES, 'foo')
        return self.assertFailure(d, ftp.PermissionDeniedError)

    def test_notDirectory(self):
        """
        C{errno.ENOTDIR} should be translated to L{ftp.IsNotADirectoryError}.
        """
        d = ftp.errnoToFailure(errno.ENOTDIR, 'foo')
        return self.assertFailure(d, ftp.IsNotADirectoryError)

    def test_fileExists(self):
        """
        C{errno.EEXIST} should be translated to L{ftp.FileExistsError}.
        """
        d = ftp.errnoToFailure(errno.EEXIST, 'foo')
        return self.assertFailure(d, ftp.FileExistsError)

    def test_isDirectory(self):
        """
        C{errno.EISDIR} should be translated to L{ftp.IsADirectoryError}.
        """
        d = ftp.errnoToFailure(errno.EISDIR, 'foo')
        return self.assertFailure(d, ftp.IsADirectoryError)

    def test_passThrough(self):
        """
        If an unknown errno is passed to L{ftp.errnoToFailure}, it should let
        the originating exception pass through.
        """
        try:
            raise RuntimeError('bar')
        except BaseException:
            d = ftp.errnoToFailure(-1, 'foo')
            return self.assertFailure(d, RuntimeError)
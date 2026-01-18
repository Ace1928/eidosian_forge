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
class IsGlobbingExpressionTests(TestCase):
    """
    Tests for _isGlobbingExpression utility function.
    """

    def test_isGlobbingExpressionEmptySegments(self):
        """
        _isGlobbingExpression will return False for None, or empty
        segments.
        """
        self.assertFalse(ftp._isGlobbingExpression())
        self.assertFalse(ftp._isGlobbingExpression([]))
        self.assertFalse(ftp._isGlobbingExpression(None))

    def test_isGlobbingExpressionNoGlob(self):
        """
        _isGlobbingExpression will return False for plain segments.

        Also, it only checks the last segment part (filename) and will not
        check the path name.
        """
        self.assertFalse(ftp._isGlobbingExpression(['ignore', 'expr']))
        self.assertFalse(ftp._isGlobbingExpression(['*.txt', 'expr']))

    def test_isGlobbingExpressionGlob(self):
        """
        _isGlobbingExpression will return True for segments which contains
        globbing characters in the last segment part (filename).
        """
        self.assertTrue(ftp._isGlobbingExpression(['ignore', '*.txt']))
        self.assertTrue(ftp._isGlobbingExpression(['ignore', '[a-b].txt']))
        self.assertTrue(ftp._isGlobbingExpression(['ignore', 'fil?.txt']))
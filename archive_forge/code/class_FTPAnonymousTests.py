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
class FTPAnonymousTests(FTPServerTestCase):
    """
    Simple tests for an FTP server with different anonymous username.
    The new anonymous username used in this test case is "guest"
    """
    userAnonymous = 'guest'

    def test_anonymousLogin(self):
        """
        Tests whether the changing of the anonymous username is working or not.
        The FTP server should not comply about the need of password for the
        username 'guest', letting it login as anonymous asking just an email
        address as password.
        """
        d = self.assertCommandResponse('USER guest', ['331 Guest login ok, type your email address as password.'])
        return self.assertCommandResponse('PASS test@twistedmatrix.com', ['230 Anonymous login ok, access restrictions apply.'], chainDeferred=d)
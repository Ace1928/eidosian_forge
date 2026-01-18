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
def _testLogin(self):
    """
        Test the login part.
        """
    self.assertEqual(self.transport.value(), b'')
    self.client.lineReceived(b'331 Guest login ok, type your email address as password.')
    self.assertEqual(self.transport.value(), b'USER anonymous\r\n')
    self.transport.clear()
    self.client.lineReceived(b'230 Anonymous login ok, access restrictions apply.')
    self.assertEqual(self.transport.value(), b'TYPE I\r\n')
    self.transport.clear()
    self.client.lineReceived(b'200 Type set to I.')
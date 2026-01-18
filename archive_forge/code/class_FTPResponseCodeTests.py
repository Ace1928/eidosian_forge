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
class FTPResponseCodeTests(TestCase):
    """
    Tests relating directly to response codes.
    """

    def test_unique(self):
        """
        All of the response code globals (for example C{RESTART_MARKER_REPLY} or
        C{USR_NAME_OK_NEED_PASS}) have unique values and are present in the
        C{RESPONSE} dictionary.
        """
        allValues = set(ftp.RESPONSE)
        seenValues = set()
        for key, value in vars(ftp).items():
            if isinstance(value, str) and key.isupper():
                self.assertIn(value, allValues, 'Code {!r} with value {!r} missing from RESPONSE dict'.format(key, value))
                self.assertNotIn(value, seenValues, f'Duplicate code {key!r} with value {value!r}')
                seenValues.add(value)
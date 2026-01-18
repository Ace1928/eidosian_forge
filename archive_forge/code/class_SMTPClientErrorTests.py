import base64
import inspect
import re
from io import BytesIO
from typing import Any, List, Optional, Tuple, Type
from zope.interface import directlyProvides, implementer
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.error
import twisted.cred.portal
from twisted import cred
from twisted.cred.checkers import AllowAnonymousAccess, ICredentialsChecker
from twisted.cred.credentials import IAnonymous
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import address, defer, error, interfaces, protocol, reactor, task
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.mail import smtp
from twisted.mail._cred import LOGINCredentials
from twisted.protocols import basic, loopback
from twisted.python.util import LineLog
from twisted.trial.unittest import TestCase
class SMTPClientErrorTests(TestCase):
    """
    Tests for L{smtp.SMTPClientError}.
    """

    def test_str(self):
        """
        The string representation of a L{SMTPClientError} instance includes
        the response code and response string.
        """
        err = smtp.SMTPClientError(123, 'some text')
        self.assertEqual(str(err), '123 some text')

    def test_strWithNegativeCode(self):
        """
        If the response code supplied to L{SMTPClientError} is negative, it
        is excluded from the string representation.
        """
        err = smtp.SMTPClientError(-1, b'foo bar')
        self.assertEqual(str(err), 'foo bar')

    def test_strWithLog(self):
        """
        If a line log is supplied to L{SMTPClientError}, its contents are
        included in the string representation of the exception instance.
        """
        log = LineLog(10)
        log.append(b'testlog')
        log.append(b'secondline')
        err = smtp.SMTPClientError(100, 'test error', log=log.str())
        self.assertEqual(str(err), '100 test error\ntestlog\nsecondline\n')
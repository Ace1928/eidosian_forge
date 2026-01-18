import base64
import hmac
import itertools
from collections import OrderedDict
from hashlib import md5
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.portal
import twisted.internet.protocol
import twisted.mail.pop3
import twisted.mail.protocols
from twisted import cred, internet, mail
from twisted.cred.credentials import IUsernameHashedPassword
from twisted.internet import defer
from twisted.internet.testing import LineSendingProtocol
from twisted.mail import pop3
from twisted.protocols import loopback
from twisted.python import failure
from twisted.trial import unittest, util
@implementer(pop3.IServerFactory)
class TestServerFactory:
    """
    A L{pop3.IServerFactory} implementation, for use by the test suite, with
    some behavior controlled by the values of (settable) public attributes and
    other behavior based on values hard-coded both here and in some test
    methods.
    """

    def cap_IMPLEMENTATION(self):
        """
        Return the hard-coded value.

        @return: L{pop3.IServerFactory}
        """
        return 'Test Implementation String'

    def cap_EXPIRE(self):
        """
        Return the hard-coded value.

        @return: L{pop3.IServerFactory}
        """
        return 60
    challengers = OrderedDict([(b'SCHEME_1', None), (b'SCHEME_2', None)])

    def cap_LOGIN_DELAY(self):
        """
        Return the hard-coded value.

        @return: L{pop3.IServerFactory}
        """
        return 120
    pue = True

    def perUserExpiration(self):
        """
        Return the hard-coded value.

        @return: L{pop3.IServerFactory}
        """
        return self.pue
    puld = True

    def perUserLoginDelay(self):
        """
        Return the hard-coded value.

        @return: L{pop3.IServerFactory}
        """
        return self.puld
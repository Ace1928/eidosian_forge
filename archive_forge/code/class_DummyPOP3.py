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
class DummyPOP3(pop3.POP3):
    """
    A simple POP3 server with a hard-coded mailbox for any user.
    """
    magic = b'<moshez>'

    def authenticateUserAPOP(self, user, password):
        """
        Succeed with a L{DummyMailbox}.

        @param user: ignored
        @param password: ignored

        @return: A three-tuple like the one returned by
            L{IRealm.requestAvatar}.
        """
        return (pop3.IMailbox, DummyMailbox(ValueError), lambda: None)
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
class MyVirtualPOP3(mail.protocols.VirtualPOP3):
    """
    A virtual-domain-supporting POP3 server.
    """
    magic = b'<moshez>'

    def authenticateUserAPOP(self, user, digest):
        """
        Authenticate against a user against a virtual domain.

        @param user: The username.
        @param digest: The digested password.

        @return: A three-tuple like the one returned by
            L{IRealm.requestAvatar}.  The mailbox will be for the user given
            by C{user}.
        """
        user, domain = self.lookupDomain(user)
        return self.service.domains[b'baz.com'].authenticateUserAPOP(user, digest, self.magic, domain)
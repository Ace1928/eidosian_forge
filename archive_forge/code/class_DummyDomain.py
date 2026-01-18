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
class DummyDomain:
    """
    A virtual domain for a POP3 server.
    """

    def __init__(self):
        self.users = {}

    def addUser(self, name):
        """
        Create a mailbox for a new user.

        @param name: The username.
        """
        self.users[name] = []

    def addMessage(self, name, message):
        """
        Add a message to the mailbox of the named user.

        @param name: The username.
        @param message: The contents of the message.
        """
        self.users[name].append(message)

    def authenticateUserAPOP(self, name, digest, magic, domain):
        """
        Succeed with a L{ListMailbox}.

        @param name: The name of the user authenticating.
        @param digest: ignored
        @param magic: ignored
        @param domain: ignored

        @return: A three-tuple like the one returned by
            L{IRealm.requestAvatar}.  The mailbox will be for the user given
            by C{name}.
        """
        return (pop3.IMailbox, ListMailbox(self.users[name]), lambda: None)
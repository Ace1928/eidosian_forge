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
class DummyMessage:
    """
    L{BrokenMessage} is an L{IMessage} which saves the message delivered to it
    to its domain object.

    @ivar domain: A L{DummyDomain} which will be used to store the message once
        it is received.
    """

    def __init__(self, domain, user):
        self.domain = domain
        self.user = user
        self.buffer = []

    def lineReceived(self, line):
        if not re.match(b'Received: From yyy.com \\(\\[.*\\]\\) by localhost;', line):
            self.buffer.append(line)

    def eomReceived(self):
        message = b'\n'.join(self.buffer) + b'\n'
        self.domain.messages[self.user.dest.local].append(message)
        deferred = defer.Deferred()
        deferred.callback(b'saved')
        return deferred
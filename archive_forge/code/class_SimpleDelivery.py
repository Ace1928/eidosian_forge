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
@implementer(smtp.IMessageDelivery)
class SimpleDelivery:
    """
    L{SimpleDelivery} is a message delivery factory with no interesting
    behavior.
    """

    def __init__(self, messageFactory):
        self._messageFactory = messageFactory

    def receivedHeader(self, helo, origin, recipients):
        return None

    def validateFrom(self, helo, origin):
        return origin

    def validateTo(self, user):
        return lambda: self._messageFactory(user)
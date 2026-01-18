import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
class OverridingLocator(TestLocator):
    """
    A locator which overrides the responder to the 'simple' command.
    """

    def greetingResponder(self, greeting, cookie):
        """
        Return a different cookieplus than L{TestLocator.greetingResponder}.
        """
        self.greetings.append((greeting, cookie))
        return dict(cookieplus=cookie + 4)
    greetingResponder = SimpleGreeting.responder(greetingResponder)
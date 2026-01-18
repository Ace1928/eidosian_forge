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
class FakeSender:
    """
    This is a fake implementation of the 'box sender' interface implied by
    L{AMP}.
    """

    def __init__(self):
        """
        Create a fake sender and initialize the list of received boxes and
        unhandled errors.
        """
        self.sentBoxes = []
        self.unhandledErrors = []
        self.expectedErrors = 0

    def expectError(self):
        """
        Expect one error, so that the test doesn't fail.
        """
        self.expectedErrors += 1

    def sendBox(self, box):
        """
        Accept a box, but don't do anything.
        """
        self.sentBoxes.append(box)

    def unhandledError(self, failure):
        """
        Deal with failures by instantly re-raising them for easier debugging.
        """
        self.expectedErrors -= 1
        if self.expectedErrors < 0:
            failure.raiseException()
        else:
            self.unhandledErrors.append(failure)
import pickle
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.application import internet
from twisted.application.internet import (
from twisted.internet import task
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.logger import formatEvent, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
@implementer(IListeningPort)
class FakePort:
    """
    Fake L{IListeningPort} implementation.

    @ivar deferred: The L{Deferred} returned by C{stopListening}.
    """
    deferred = None

    def stopListening(self):
        """
        Stop listening.

        @return: a L{Deferred} stored in L{FakePort.deferred}
        """
        self.deferred = Deferred()
        return self.deferred

    def getHost(self):
        pass

    def startListening(self):
        pass
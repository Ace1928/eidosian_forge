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
class ConnectInformation:
    """
    Information about C{endpointForTesting}

    @ivar connectQueue: a L{list} of L{Deferred} returned from C{connect}.  If
        these are not already fired, you can fire them with no value and they
        will trigger building a factory.

    @ivar constructedProtocols: a L{list} of protocols constructed.

    @ivar passedFactories: a L{list} of L{IProtocolFactory}; the ones actually
        passed to the underlying endpoint / i.e. the reactor.
    """

    def __init__(self):
        self.connectQueue = []
        self.constructedProtocols = []
        self.passedFactories = []
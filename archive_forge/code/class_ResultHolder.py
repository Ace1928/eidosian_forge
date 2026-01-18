from collections import defaultdict
from socket import (
from threading import Lock, local
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted._threads import LockWorker, Team, createMemoryWorker
from twisted.internet._resolver import (
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.base import PluggableResolverMixin, ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import (
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import SynchronousTestCase as UnitTest
@implementer(IResolutionReceiver)
class ResultHolder:
    """
    A resolution receiver which holds onto the results it received.
    """
    _started = False
    _ended = False

    def __init__(self, testCase):
        """
        Create a L{ResultHolder} with a L{UnitTest}.
        """
        self._testCase = testCase

    def resolutionBegan(self, hostResolution):
        """
        Hostname resolution began.

        @param hostResolution: see L{IResolutionReceiver}
        """
        self._started = True
        self._resolution = hostResolution
        self._addresses = []

    def addressResolved(self, address):
        """
        An address was resolved.

        @param address: see L{IResolutionReceiver}
        """
        self._addresses.append(address)

    def resolutionComplete(self):
        """
        Hostname resolution is complete.
        """
        self._ended = True
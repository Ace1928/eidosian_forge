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
def _resolveOnlyTest(self, addrTypes, expectedAF):
    """
        Verify that the given set of address types results in the given C{AF_}
        constant being passed to C{getaddrinfo}.

        @param addrTypes: iterable of L{IAddress} implementers

        @param expectedAF: an C{AF_*} constant
        """
    receiver = ResultHolder(self)
    resolution = self.resolver.resolveHostName(receiver, 'sample.example.com', addressTypes=addrTypes)
    self.assertIs(receiver._resolution, resolution)
    self.doThreadWork()
    self.doReactorWork()
    host, port, family, socktype, proto, flags = self.getter.calls[0]
    self.assertEqual(family, expectedAF)
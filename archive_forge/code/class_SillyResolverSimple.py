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
@implementer(IResolverSimple)
class SillyResolverSimple:
    """
    Trivial implementation of L{IResolverSimple}
    """

    def __init__(self):
        """
        Create a L{SillyResolverSimple} with a queue of requests it is working
        on.
        """
        self._requests = []

    def getHostByName(self, name, timeout=()):
        """
        Implement L{IResolverSimple.getHostByName}.

        @param name: see L{IResolverSimple.getHostByName}.

        @param timeout: see L{IResolverSimple.getHostByName}.

        @return: see L{IResolverSimple.getHostByName}.
        """
        self._requests.append(Deferred())
        return self._requests[-1]
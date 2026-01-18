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
def addResultForHost(self, host, sockaddr, family=AF_INET, socktype=SOCK_STREAM, proto=IPPROTO_TCP, canonname=b''):
    """
        Add a result for a given hostname.  When this hostname is resolved, the
        result will be a L{list} of all results C{addResultForHost} has been
        called with using that hostname so far.

        @param host: The hostname to give this result for.  This will be the
            next result from L{FakeAddrInfoGetter.getaddrinfo} when passed this
            host.

        @type canonname: native L{str}

        @param sockaddr: The resulting socket address; should be a 2-tuple for
            IPv4 or a 4-tuple for IPv6.

        @param family: An C{AF_*} constant that will be returned from
            C{getaddrinfo}.

        @param socktype: A C{SOCK_*} constant that will be returned from
            C{getaddrinfo}.

        @param proto: An C{IPPROTO_*} constant that will be returned from
            C{getaddrinfo}.

        @param canonname: A canonical name that will be returned from
            C{getaddrinfo}.
        @type canonname: native L{str}
        """
    self.results[host].append((family, socktype, proto, canonname, sockaddr))
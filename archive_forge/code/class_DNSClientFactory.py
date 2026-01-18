import errno
import os
import warnings
from zope.interface import moduleProvides
from twisted.internet import defer, error, interfaces, protocol
from twisted.internet.abstract import isIPv6Address
from twisted.names import cache, common, dns, hosts as hostsModule, resolve, root
from twisted.python import failure, log
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.internet.base import ThreadedResolver as _ThreadedResolverImpl
class DNSClientFactory(protocol.ClientFactory):

    def __init__(self, controller, timeout=10):
        self.controller = controller
        self.timeout = timeout

    def clientConnectionLost(self, connector, reason):
        pass

    def clientConnectionFailed(self, connector, reason):
        """
        Fail all pending TCP DNS queries if the TCP connection attempt
        fails.

        @see: L{twisted.internet.protocol.ClientFactory}

        @param connector: Not used.
        @type connector: L{twisted.internet.interfaces.IConnector}

        @param reason: A C{Failure} containing information about the
            cause of the connection failure. This will be passed as the
            argument to C{errback} on every pending TCP query
            C{deferred}.
        @type reason: L{twisted.python.failure.Failure}
        """
        pending = self.controller.pending[:]
        del self.controller.pending[:]
        for pendingState in pending:
            d = pendingState[0]
            d.errback(reason)

    def buildProtocol(self, addr):
        p = dns.DNSProtocol(self.controller)
        p.factory = self
        return p
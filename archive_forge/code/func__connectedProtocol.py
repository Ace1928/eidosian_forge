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
def _connectedProtocol(self, interface=''):
    """
        Return a new L{DNSDatagramProtocol} bound to a randomly selected port
        number.
        """
    failures = 0
    proto = dns.DNSDatagramProtocol(self, reactor=self._reactor)
    while True:
        try:
            self._reactor.listenUDP(dns.randomSource(), proto, interface=interface)
        except error.CannotListenError as e:
            failures += 1
            if hasattr(e.socketError, 'errno') and e.socketError.errno == errno.EMFILE:
                raise
            if failures >= 1000:
                raise
        else:
            return proto
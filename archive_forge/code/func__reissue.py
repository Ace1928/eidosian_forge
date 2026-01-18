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
def _reissue(self, reason, addressesLeft, addressesUsed, query, timeout):
    reason.trap(dns.DNSQueryTimeoutError)
    if not addressesLeft:
        addressesLeft = addressesUsed
        addressesLeft.reverse()
        addressesUsed = []
        timeout = timeout[1:]
    if not timeout:
        return failure.Failure(defer.TimeoutError(query))
    address = addressesLeft.pop()
    addressesUsed.append(address)
    d = self._query(address, query, timeout[0], reason.value.id)
    d.addErrback(self._reissue, addressesLeft, addressesUsed, query, timeout)
    return d
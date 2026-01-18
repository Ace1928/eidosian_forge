from twisted.internet import defer
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.names import common, dns
from twisted.python import failure
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
def _aRecords(self, name):
    """
        Return a tuple of L{dns.RRHeader} instances for all of the IPv4
        addresses in the hosts file.
        """
    return tuple((dns.RRHeader(name, dns.A, dns.IN, self.ttl, dns.Record_A(addr, self.ttl)) for addr in searchFileForAll(FilePath(self.file), name) if isIPAddress(addr)))
import random
from zope.interface import implementer
from twisted.internet import error, interfaces
from twisted.names import client, dns
from twisted.names.error import DNSNameError
from twisted.python.compat import nativeString
def _ebServiceUnknown(self, failure):
    """
        Connect to the default port when the service name is unknown.

        If no SRV records were found, the service name will be passed as the
        port. If resolving the name fails with
        L{error.ServiceNameUnknownError}, a final attempt is done using the
        default port.
        """
    failure.trap(error.ServiceNameUnknownError)
    self.servers = [dns.Record_SRV(0, 0, self._defaultPort, self.domain)]
    self.orderedServers = []
    self.connect()
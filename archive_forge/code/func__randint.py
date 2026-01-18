import random
from zope.interface.verify import verifyObject
from twisted.internet import defer, protocol
from twisted.internet.error import DNSLookupError, ServiceNameUnknownError
from twisted.internet.interfaces import IConnector
from twisted.internet.testing import MemoryReactor
from twisted.names import client, dns, srvconnect
from twisted.names.common import ResolverBase
from twisted.names.error import DNSNameError
from twisted.trial import unittest
def _randint(self, min, max):
    """
        Fake randint.

        Returns the first element of L{randIntResults} and records the
        arguments passed to it in L{randIntArgs}.

        @param min: Lower bound of the random number.
        @type min: L{int}

        @param max: Higher bound of the random number.
        @type max: L{int}

        @return: Fake random number from L{randIntResults}.
        @rtype: L{int}
        """
    self.randIntArgs.append((min, max))
    return self.randIntResults.pop(0)
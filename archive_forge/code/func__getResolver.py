from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.internet.defer import Deferred, TimeoutError, gatherResults, succeed
from twisted.internet.interfaces import IResolverSimple
from twisted.names import client, root
from twisted.names.dns import (
from twisted.names.error import DNSNameError, ResolverError
from twisted.names.root import Resolver
from twisted.names.test.test_util import MemoryReactor
from twisted.python.log import msg
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase, TestCase
def _getResolver(self, serverResponses, maximumQueries=10):
    """
        Create and return a new L{root.Resolver} modified to resolve queries
        against the record data represented by C{servers}.

        @param serverResponses: A mapping from dns server addresses to
            mappings.  The inner mappings are from query two-tuples (name,
            type) to dictionaries suitable for use as **arguments to
            L{_respond}.  See that method for details.
        """
    roots = ['1.1.2.3']
    resolver = Resolver(roots, maximumQueries)

    def query(query, serverAddresses, timeout, filter):
        msg(f'Query for QNAME {query.name} at {serverAddresses!r}')
        for addr in serverAddresses:
            try:
                server = serverResponses[addr]
            except KeyError:
                continue
            records = server[query.name.name, query.type]
            return succeed(self._respond(**records))
    resolver._query = query
    return resolver
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
def _queryTest(self, filter):
    """
        Invoke L{Resolver._query} and verify that it sends the correct DNS
        query.  Deliver a canned response to the query and return whatever the
        L{Deferred} returned by L{Resolver._query} fires with.

        @param filter: The value to pass for the C{filter} parameter to
            L{Resolver._query}.
        """
    reactor = MemoryReactor()
    resolver = Resolver([], reactor=reactor)
    d = resolver._query(Query(b'foo.example.com', A, IN), [('1.1.2.3', 1053)], (30,), filter)
    portNumber, transport = reactor.udpPorts.popitem()
    [(packet, address)] = transport._sentPackets
    message = Message()
    message.fromStr(packet)
    self.assertEqual(message.queries, [Query(b'foo.example.com', A, IN)])
    self.assertEqual(message.answers, [])
    self.assertEqual(message.authority, [])
    self.assertEqual(message.additional, [])
    response = []
    d.addCallback(response.append)
    self.assertEqual(response, [])
    del message.queries[:]
    message.answer = 1
    message.answers.append(RRHeader(b'foo.example.com', payload=Record_A('5.8.13.21')))
    transport._protocol.datagramReceived(message.toStr(), ('1.1.2.3', 1053))
    return response[0]
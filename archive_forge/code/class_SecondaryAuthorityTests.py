import copy
import operator
import socket
from functools import partial, reduce
from io import BytesIO
from struct import pack
from twisted.internet import defer, error, reactor
from twisted.internet.defer import succeed
from twisted.internet.testing import (
from twisted.names import authority, client, common, dns, server
from twisted.names.client import Resolver
from twisted.names.dns import SOA, Message, Query, Record_A, Record_SOA, RRHeader
from twisted.names.error import DomainError
from twisted.names.secondary import SecondaryAuthority, SecondaryAuthorityService
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.trial import unittest
class SecondaryAuthorityTests(unittest.TestCase):
    """
    L{twisted.names.secondary.SecondaryAuthority} correctly constructs objects
    with a specified IP address and optionally specified DNS port.
    """

    def test_defaultPort(self):
        """
        When constructed using L{SecondaryAuthority.__init__}, the default port
        of 53 is used.
        """
        secondary = SecondaryAuthority('192.168.1.1', 'inside.com')
        self.assertEqual(secondary.primary, '192.168.1.1')
        self.assertEqual(secondary._port, 53)
        self.assertEqual(secondary.domain, b'inside.com')

    def test_explicitPort(self):
        """
        When constructed using L{SecondaryAuthority.fromServerAddressAndDomain},
        the specified port is used.
        """
        secondary = SecondaryAuthority.fromServerAddressAndDomain(('192.168.1.1', 5353), 'inside.com')
        self.assertEqual(secondary.primary, '192.168.1.1')
        self.assertEqual(secondary._port, 5353)
        self.assertEqual(secondary.domain, b'inside.com')

    def test_transfer(self):
        """
        An attempt is made to transfer the zone for the domain the
        L{SecondaryAuthority} was constructed with from the server address it
        was constructed with when L{SecondaryAuthority.transfer} is called.
        """
        secondary = SecondaryAuthority.fromServerAddressAndDomain(('192.168.1.2', 1234), 'example.com')
        secondary._reactor = reactor = MemoryReactorClock()
        secondary.transfer()
        host, port, factory, timeout, bindAddress = reactor.tcpClients.pop(0)
        self.assertEqual(host, '192.168.1.2')
        self.assertEqual(port, 1234)
        proto = factory.buildProtocol((host, port))
        transport = StringTransport()
        proto.makeConnection(transport)
        msg = Message()
        msg.decode(BytesIO(transport.value()[2:]))
        self.assertEqual([dns.Query('example.com', dns.AXFR, dns.IN)], msg.queries)

    def test_lookupAddress(self):
        """
        L{SecondaryAuthority.lookupAddress} returns a L{Deferred} that fires
        with the I{A} records the authority has cached from the primary.
        """
        secondary = SecondaryAuthority.fromServerAddressAndDomain(('192.168.1.2', 1234), b'example.com')
        secondary._reactor = reactor = MemoryReactorClock()
        secondary.transfer()
        host, port, factory, timeout, bindAddress = reactor.tcpClients.pop(0)
        proto = factory.buildProtocol((host, port))
        transport = StringTransport()
        proto.makeConnection(transport)
        query = Message(answer=1, auth=1)
        query.decode(BytesIO(transport.value()[2:]))
        soa = Record_SOA(mname=b'ns1.example.com', rname='admin.example.com', serial=123456, refresh=3600, minimum=4800, expire=7200, retry=9600, ttl=12000)
        a = Record_A(b'192.168.1.2', ttl=0)
        answer = Message(id=query.id, answer=1, auth=1)
        answer.answers.extend([RRHeader(b'example.com', type=SOA, payload=soa), RRHeader(b'example.com', payload=a), RRHeader(b'example.com', type=SOA, payload=soa)])
        data = answer.toStr()
        proto.dataReceived(pack('!H', len(data)) + data)
        result = self.successResultOf(secondary.lookupAddress('example.com'))
        self.assertEqual(([RRHeader(b'example.com', payload=a, auth=True)], [], []), result)
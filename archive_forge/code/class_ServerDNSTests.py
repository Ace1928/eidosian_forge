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
class ServerDNSTests(unittest.TestCase):
    """
    Test cases for DNS server and client.
    """

    def setUp(self):
        self.factory = server.DNSServerFactory([test_domain_com, reverse_domain, my_domain_com], verbose=2)
        p = dns.DNSDatagramProtocol(self.factory)
        while 1:
            listenerTCP = reactor.listenTCP(0, self.factory, interface='127.0.0.1')
            self.addCleanup(listenerTCP.stopListening)
            port = listenerTCP.getHost().port
            try:
                listenerUDP = reactor.listenUDP(port, p, interface='127.0.0.1')
            except error.CannotListenError:
                pass
            else:
                self.addCleanup(listenerUDP.stopListening)
                break
        self.listenerTCP = listenerTCP
        self.listenerUDP = listenerUDP
        self.resolver = client.Resolver(servers=[('127.0.0.1', port)])

    def tearDown(self):
        """
        Clean up any server connections associated with the
        L{DNSServerFactory} created in L{setUp}
        """
        for conn in self.factory.connections[:]:
            conn.transport.loseConnection()
        return waitUntilAllDisconnected(reactor, self.factory.connections[:])

    def namesTest(self, querying, expectedRecords):
        """
        Assert that the DNS response C{querying} will eventually fire with
        contains exactly a certain collection of records.

        @param querying: A L{Deferred} returned from one of the DNS client
            I{lookup} methods.

        @param expectedRecords: A L{list} of L{IRecord} providers which must be
            in the response or the test will be failed.

        @return: A L{Deferred} that fires when the assertion has been made.  It
            fires with a success result if the assertion succeeds and with a
            L{Failure} if it fails.
        """

        def checkResults(response):
            receivedRecords = justPayload(response)
            self.assertEqual(set(expectedRecords), set(receivedRecords))
        querying.addCallback(checkResults)
        return querying

    def test_addressRecord1(self):
        """Test simple DNS 'A' record queries"""
        return self.namesTest(self.resolver.lookupAddress('test-domain.com'), [dns.Record_A('127.0.0.1', ttl=19283784)])

    def test_addressRecord2(self):
        """Test DNS 'A' record queries with multiple answers"""
        return self.namesTest(self.resolver.lookupAddress('host.test-domain.com'), [dns.Record_A('123.242.1.5', ttl=19283784), dns.Record_A('0.255.0.255', ttl=19283784)])

    def test_addressRecord3(self):
        """Test DNS 'A' record queries with edge cases"""
        return self.namesTest(self.resolver.lookupAddress('host-two.test-domain.com'), [dns.Record_A('255.255.255.254', ttl=19283784), dns.Record_A('0.0.0.0', ttl=19283784)])

    def test_authority(self):
        """Test DNS 'SOA' record queries"""
        return self.namesTest(self.resolver.lookupAuthority('test-domain.com'), [soa_record])

    def test_mailExchangeRecord(self):
        """
        The DNS client can issue an MX query and receive a response including
        an MX record as well as any A record hints.
        """
        return self.namesTest(self.resolver.lookupMailExchange(b'test-domain.com'), [dns.Record_MX(10, b'host.test-domain.com', ttl=19283784), dns.Record_A(b'123.242.1.5', ttl=19283784), dns.Record_A(b'0.255.0.255', ttl=19283784)])

    def test_nameserver(self):
        """Test DNS 'NS' record queries"""
        return self.namesTest(self.resolver.lookupNameservers('test-domain.com'), [dns.Record_NS('39.28.189.39', ttl=19283784)])

    def test_HINFO(self):
        """Test DNS 'HINFO' record queries"""
        return self.namesTest(self.resolver.lookupHostInfo('test-domain.com'), [dns.Record_HINFO(os=b'Linux', cpu=b'A Fast One, Dontcha know', ttl=19283784)])

    def test_PTR(self):
        """Test DNS 'PTR' record queries"""
        return self.namesTest(self.resolver.lookupPointer('123.93.84.28.in-addr.arpa'), [dns.Record_PTR('test.host-reverse.lookup.com', ttl=11193983)])

    def test_CNAME(self):
        """Test DNS 'CNAME' record queries"""
        return self.namesTest(self.resolver.lookupCanonicalName('test-domain.com'), [dns.Record_CNAME('canonical.name.com', ttl=19283784)])

    def test_MB(self):
        """Test DNS 'MB' record queries"""
        return self.namesTest(self.resolver.lookupMailBox('test-domain.com'), [dns.Record_MB('mailbox.test-domain.com', ttl=19283784)])

    def test_MG(self):
        """Test DNS 'MG' record queries"""
        return self.namesTest(self.resolver.lookupMailGroup('test-domain.com'), [dns.Record_MG('mail.group.someplace', ttl=19283784)])

    def test_MR(self):
        """Test DNS 'MR' record queries"""
        return self.namesTest(self.resolver.lookupMailRename('test-domain.com'), [dns.Record_MR('mail.redirect.or.whatever', ttl=19283784)])

    def test_MINFO(self):
        """Test DNS 'MINFO' record queries"""
        return self.namesTest(self.resolver.lookupMailboxInfo('test-domain.com'), [dns.Record_MINFO(rmailbx='r mail box', emailbx='e mail box', ttl=19283784)])

    def test_SRV(self):
        """Test DNS 'SRV' record queries"""
        return self.namesTest(self.resolver.lookupService('http.tcp.test-domain.com'), [dns.Record_SRV(257, 16383, 43690, 'some.other.place.fool', ttl=19283784)])

    def test_AFSDB(self):
        """Test DNS 'AFSDB' record queries"""
        return self.namesTest(self.resolver.lookupAFSDatabase('test-domain.com'), [dns.Record_AFSDB(subtype=1, hostname='afsdb.test-domain.com', ttl=19283784)])

    def test_RP(self):
        """Test DNS 'RP' record queries"""
        return self.namesTest(self.resolver.lookupResponsibility('test-domain.com'), [dns.Record_RP(mbox='whatever.i.dunno', txt='some.more.text', ttl=19283784)])

    def test_TXT(self):
        """Test DNS 'TXT' record queries"""
        return self.namesTest(self.resolver.lookupText('test-domain.com'), [dns.Record_TXT(b'A First piece of Text', b'a SecoNd piece', ttl=19283784), dns.Record_TXT(b'Some more text, haha!  Yes.  \x00  Still here?', ttl=19283784)])

    def test_spf(self):
        """
        L{DNSServerFactory} can serve I{SPF} resource records.
        """
        return self.namesTest(self.resolver.lookupSenderPolicy('test-domain.com'), [dns.Record_SPF(b'v=spf1 mx/30 mx:example.org/30 -all', ttl=19283784), dns.Record_SPF(b'v=spf1 +mx a:\x00colo', b'.example.com/28 -all not valid', ttl=19283784)])

    def test_WKS(self):
        """Test DNS 'WKS' record queries"""
        return self.namesTest(self.resolver.lookupWellKnownServices('test-domain.com'), [dns.Record_WKS('12.54.78.12', socket.IPPROTO_TCP, b'\x12\x01\x16\xfe\xc1\x00\x01', ttl=19283784)])

    def test_someRecordsWithTTLs(self):
        result_soa = copy.copy(my_soa)
        result_soa.ttl = my_soa.expire
        return self.namesTest(self.resolver.lookupAllRecords('my-domain.com'), [result_soa, dns.Record_A('1.2.3.4', ttl='1S'), dns.Record_NS('ns1.domain', ttl='2M'), dns.Record_NS('ns2.domain', ttl='3H'), dns.Record_SRV(257, 16383, 43690, 'some.other.place.fool', ttl='4D')])

    def test_AAAA(self):
        """Test DNS 'AAAA' record queries (IPv6)"""
        return self.namesTest(self.resolver.lookupIPV6Address('test-domain.com'), [dns.Record_AAAA('AF43:5634:1294:AFCB:56AC:48EF:34C3:01FF', ttl=19283784)])

    def test_A6(self):
        """Test DNS 'A6' record queries (IPv6)"""
        return self.namesTest(self.resolver.lookupAddress6('test-domain.com'), [dns.Record_A6(0, 'ABCD::4321', '', ttl=19283784), dns.Record_A6(12, '0:0069::0', 'some.network.tld', ttl=19283784), dns.Record_A6(8, '0:5634:1294:AFCB:56AC:48EF:34C3:01FF', 'tra.la.la.net', ttl=19283784)])

    def test_zoneTransfer(self):
        """
        Test DNS 'AXFR' queries (Zone transfer)
        """
        default_ttl = soa_record.expire
        results = [copy.copy(r) for r in reduce(operator.add, test_domain_com.records.values())]
        for r in results:
            if r.ttl is None:
                r.ttl = default_ttl
        return self.namesTest(self.resolver.lookupZone('test-domain.com').addCallback(lambda r: (r[0][:-1],)), results)

    def test_zoneTransferConnectionFails(self):
        """
        A failed AXFR TCP connection errbacks the L{Deferred} returned
        from L{Resolver.lookupZone}.
        """
        resolver = Resolver(servers=[('nameserver.invalid', 53)])
        return self.assertFailure(resolver.lookupZone('impossible.invalid'), error.DNSLookupError)

    def test_similarZonesDontInterfere(self):
        """Tests that unrelated zones don't mess with each other."""
        return self.namesTest(self.resolver.lookupAddress('anothertest-domain.com'), [dns.Record_A('1.2.3.4', ttl=19283784)])

    def test_NAPTR(self):
        """
        Test DNS 'NAPTR' record queries.
        """
        return self.namesTest(self.resolver.lookupNamingAuthorityPointer('test-domain.com'), [dns.Record_NAPTR(100, 10, b'u', b'sip+E2U', b'!^.*$!sip:information@domain.tld!', ttl=19283784)])
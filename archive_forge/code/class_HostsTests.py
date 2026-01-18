from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
class HostsTests(SynchronousTestCase, GoodTempPathMixin):
    """
    Tests for the I{hosts(5)}-based L{twisted.names.hosts.Resolver}.
    """

    def setUp(self) -> None:
        f = self.path()
        f.setContent(b'\n1.1.1.1    EXAMPLE EXAMPLE.EXAMPLETHING\n::2        mixed\n1.1.1.2    MIXED\n::1        ip6thingy\n1.1.1.3    multiple\n1.1.1.4    multiple\n::3        ip6-multiple\n::4        ip6-multiple\nnot-an-ip  malformed\nmalformed\n# malformed\n1.1.1.5    malformed\n::5        malformed\n')
        self.ttl = 4200
        self.resolver = Resolver(f.path, self.ttl)

    def test_defaultPath(self) -> None:
        """
        The default hosts file used by L{Resolver} is I{/etc/hosts} if no value
        is given for the C{file} initializer parameter.
        """
        resolver = Resolver()
        self.assertEqual(b'/etc/hosts', resolver.file)

    def test_getHostByName(self) -> None:
        """
        L{hosts.Resolver.getHostByName} returns a L{Deferred} which fires with a
        string giving the address of the queried name as found in the resolver's
        hosts file.
        """
        data = [(b'EXAMPLE', '1.1.1.1'), (b'EXAMPLE.EXAMPLETHING', '1.1.1.1'), (b'MIXED', '1.1.1.2')]
        ds = [self.resolver.getHostByName(n).addCallback(self.assertEqual, ip) for n, ip in data]
        self.successResultOf(gatherResults(ds))

    def test_lookupAddress(self) -> None:
        """
        L{hosts.Resolver.lookupAddress} returns a L{Deferred} which fires with A
        records from the hosts file.
        """
        d = self.resolver.lookupAddress(b'multiple')
        answers, authority, additional = self.successResultOf(d)
        self.assertEqual((RRHeader(b'multiple', A, IN, self.ttl, Record_A('1.1.1.3', self.ttl)), RRHeader(b'multiple', A, IN, self.ttl, Record_A('1.1.1.4', self.ttl))), answers)

    def test_lookupIPV6Address(self) -> None:
        """
        L{hosts.Resolver.lookupIPV6Address} returns a L{Deferred} which fires
        with AAAA records from the hosts file.
        """
        d = self.resolver.lookupIPV6Address(b'ip6-multiple')
        answers, authority, additional = self.successResultOf(d)
        self.assertEqual((RRHeader(b'ip6-multiple', AAAA, IN, self.ttl, Record_AAAA('::3', self.ttl)), RRHeader(b'ip6-multiple', AAAA, IN, self.ttl, Record_AAAA('::4', self.ttl))), answers)

    def test_lookupAllRecords(self) -> None:
        """
        L{hosts.Resolver.lookupAllRecords} returns a L{Deferred} which fires
        with A records from the hosts file.
        """
        d = self.resolver.lookupAllRecords(b'mixed')
        answers, authority, additional = self.successResultOf(d)
        self.assertEqual((RRHeader(b'mixed', A, IN, self.ttl, Record_A('1.1.1.2', self.ttl)),), answers)

    def test_notImplemented(self) -> None:
        """
        L{hosts.Resolver} fails with L{NotImplementedError} for L{IResolver}
        methods it doesn't implement.
        """
        self.failureResultOf(self.resolver.lookupMailExchange(b'EXAMPLE'), NotImplementedError)

    def test_query(self) -> None:
        d = self.resolver.query(Query(b'EXAMPLE'))
        [answer], authority, additional = self.successResultOf(d)
        self.assertEqual(answer.payload.dottedQuad(), '1.1.1.1')

    def test_lookupAddressNotFound(self) -> None:
        """
        L{hosts.Resolver.lookupAddress} returns a L{Deferred} which fires with
        L{dns.DomainError} if the name passed in has no addresses in the hosts
        file.
        """
        self.failureResultOf(self.resolver.lookupAddress(b'foueoa'), DomainError)

    def test_lookupIPV6AddressNotFound(self) -> None:
        """
        Like L{test_lookupAddressNotFound}, but for
        L{hosts.Resolver.lookupIPV6Address}.
        """
        self.failureResultOf(self.resolver.lookupIPV6Address(b'foueoa'), DomainError)

    def test_lookupAllRecordsNotFound(self) -> None:
        """
        Like L{test_lookupAddressNotFound}, but for
        L{hosts.Resolver.lookupAllRecords}.
        """
        self.failureResultOf(self.resolver.lookupAllRecords(b'foueoa'), DomainError)

    def test_lookupMalformed(self) -> None:
        """
        L{hosts.Resolver.lookupAddress} returns a L{Deferred} which fires with
        the valid addresses from the hosts file, ignoring any entries that
        aren't valid IP addresses.
        """
        d = self.resolver.lookupAddress(b'malformed')
        [answer], authority, additional = self.successResultOf(d)
        self.assertEqual(RRHeader(b'malformed', A, IN, self.ttl, Record_A('1.1.1.5', self.ttl)), answer)

    def test_lookupIPV6Malformed(self) -> None:
        """
        Like L{test_lookupAddressMalformed}, but for
        L{hosts.Resolver.lookupIPV6Address}.
        """
        d = self.resolver.lookupIPV6Address(b'malformed')
        [answer], authority, additional = self.successResultOf(d)
        self.assertEqual(RRHeader(b'malformed', AAAA, IN, self.ttl, Record_AAAA('::5', self.ttl)), answer)
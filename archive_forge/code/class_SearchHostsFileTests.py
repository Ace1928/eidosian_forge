from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
class SearchHostsFileTests(SynchronousTestCase, GoodTempPathMixin):
    """
    Tests for L{searchFileFor}, a helper which finds the first address for a
    particular hostname in a I{hosts(5)}-style file.
    """

    def test_findAddress(self) -> None:
        """
        If there is an IPv4 address for the hostname passed to L{searchFileFor},
        it is returned.
        """
        hosts = self.path()
        hosts.setContent(b'10.2.3.4 foo.example.com\n')
        self.assertEqual('10.2.3.4', searchFileFor(hosts.path, b'foo.example.com'))

    def test_notFoundAddress(self) -> None:
        """
        If there is no address information for the hostname passed to
        L{searchFileFor}, L{None} is returned.
        """
        hosts = self.path()
        hosts.setContent(b'10.2.3.4 foo.example.com\n')
        self.assertIsNone(searchFileFor(hosts.path, b'bar.example.com'))

    def test_firstAddress(self) -> None:
        """
        The first address associated with the given hostname is returned.
        """
        hosts = self.path()
        hosts.setContent(b'::1 foo.example.com\n10.1.2.3 foo.example.com\nfe80::21b:fcff:feee:5a1d foo.example.com\n')
        self.assertEqual('::1', searchFileFor(hosts.path, b'foo.example.com'))

    def test_searchFileForAliases(self) -> None:
        """
        For a host with a canonical name and one or more aliases,
        L{searchFileFor} can find an address given any of the names.
        """
        hosts = self.path()
        hosts.setContent(b'127.0.1.1\thelmut.example.org\thelmut\n# a comment\n::1 localhost ip6-localhost ip6-loopback\n')
        self.assertEqual(searchFileFor(hosts.path, b'helmut'), '127.0.1.1')
        self.assertEqual(searchFileFor(hosts.path, b'helmut.example.org'), '127.0.1.1')
        self.assertEqual(searchFileFor(hosts.path, b'ip6-localhost'), '::1')
        self.assertEqual(searchFileFor(hosts.path, b'ip6-loopback'), '::1')
        self.assertEqual(searchFileFor(hosts.path, b'localhost'), '::1')
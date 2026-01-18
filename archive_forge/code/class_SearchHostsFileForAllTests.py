from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
class SearchHostsFileForAllTests(SynchronousTestCase, GoodTempPathMixin):
    """
    Tests for L{searchFileForAll}, a helper which finds all addresses for a
    particular hostname in a I{hosts(5)}-style file.
    """

    def test_allAddresses(self) -> None:
        """
        L{searchFileForAll} returns a list of all addresses associated with the
        name passed to it.
        """
        hosts = self.path()
        hosts.setContent(b'127.0.0.1     foobar.example.com\n127.0.0.2     foobar.example.com\n::1           foobar.example.com\n')
        self.assertEqual(['127.0.0.1', '127.0.0.2', '::1'], searchFileForAll(hosts, b'foobar.example.com'))

    def test_caseInsensitively(self) -> None:
        """
        L{searchFileForAll} searches for names case-insensitively.
        """
        hosts = self.path()
        hosts.setContent(b'127.0.0.1     foobar.EXAMPLE.com\n')
        self.assertEqual(['127.0.0.1'], searchFileForAll(hosts, b'FOOBAR.example.com'))

    def test_readError(self) -> None:
        """
        If there is an error reading the contents of the hosts file,
        L{searchFileForAll} returns an empty list.
        """
        self.assertEqual([], searchFileForAll(self.path(), b'example.com'))

    def test_malformedIP(self) -> None:
        """
        L{searchFileForAll} ignores any malformed IP addresses associated with
        the name passed to it.
        """
        hosts = self.path()
        hosts.setContent(b'127.0.0.1\tmiser.example.org\tmiser\nnot-an-ip\tmiser\n\xffnot-ascii\t miser\n# miser\nmiser\n::1 miser')
        self.assertEqual(['127.0.0.1', '::1'], searchFileForAll(hosts, b'miser'))
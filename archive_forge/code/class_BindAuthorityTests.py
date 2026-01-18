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
class BindAuthorityTests(unittest.TestCase):
    """
    Tests for L{twisted.names.authority.BindAuthority}.
    """

    def loadBindString(self, s):
        """
        Create a new L{twisted.names.authority.BindAuthority} from C{s}.

        @param s: A string with BIND zone data.
        @type s: bytes

        @return: a new bind authority
        @rtype: L{twisted.names.authority.BindAuthority}
        """
        fp = FilePath(self.mktemp().encode('ascii'))
        fp.setContent(s)
        return authority.BindAuthority(fp.path)

    def setUp(self):
        self.auth = self.loadBindString(sampleBindZone)

    def test_ttl(self):
        """
        Loads the default $TTL and applies it to all records.
        """
        for dom in self.auth.records.keys():
            for rec in self.auth.records[dom]:
                self.assertTrue(604800 == rec.ttl)

    def test_originFromFile(self):
        """
        Loads the default $ORIGIN.
        """
        self.assertEqual(b'example.com.', self.auth.origin)
        self.assertIn(b'not-fqdn.example.com', self.auth.records)

    def test_aRecords(self):
        """
        A records are loaded.
        """
        for dom, ip in [(b'example.com', '10.0.0.1'), (b'no-in.example.com', '10.0.0.2')]:
            [[rr], [], []] = self.successResultOf(self.auth.lookupAddress(dom))
            self.assertEqual(dns.Record_A(ip, 604800), rr.payload)

    def test_aaaaRecords(self):
        """
        AAAA records are loaded.
        """
        [[rr], [], []] = self.successResultOf(self.auth.lookupIPV6Address(b'example.com'))
        self.assertEqual(dns.Record_AAAA('2001:db8:10::1', 604800), rr.payload)

    def test_mxRecords(self):
        """
        MX records are loaded.
        """
        [[rr], [], []] = self.successResultOf(self.auth.lookupMailExchange(b'not-fqdn.example.com'))
        self.assertEqual(dns.Record_MX(preference=10, name='mx.example.com', ttl=604800), rr.payload)

    def test_cnameRecords(self):
        """
        CNAME records are loaded.
        """
        [answers, [], []] = self.successResultOf(self.auth.lookupIPV6Address(b'www.example.com'))
        rr = answers[0]
        self.assertEqual(dns.Record_CNAME(name='example.com', ttl=604800), rr.payload)

    def test_invalidRecordClass(self):
        """
        loadBindString raises NotImplementedError on invalid records.
        """
        with self.assertRaises(NotImplementedError) as e:
            self.loadBindString(b'example.com. IN LOL 192.168.0.1')
        self.assertEqual("Record type 'LOL' not supported", e.exception.args[0])

    def test_invalidDirectives(self):
        """
        $INCLUDE and $GENERATE raise NotImplementedError.
        """
        for directive in (b'$INCLUDE', b'$GENERATE'):
            with self.assertRaises(NotImplementedError) as e:
                self.loadBindString(directive + b' doesNotMatter')
            self.assertEqual(nativeString(directive + b' directive not implemented'), e.exception.args[0])
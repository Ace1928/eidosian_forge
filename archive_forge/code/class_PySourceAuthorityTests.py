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
class PySourceAuthorityTests(unittest.TestCase):
    """
    Tests for L{twisted.names.authority.PySourceAuthority}.
    """

    def loadPySourceString(self, s):
        """
        Create a new L{twisted.names.authority.PySourceAuthority} from C{s}.

        @param s: A string with BIND zone data in a Python source file.
        @type s: L{str}

        @return: a new bind authority
        @rtype: L{twisted.names.authority.PySourceAuthority}
        """
        fp = FilePath(self.mktemp())
        with open(fp.path, 'w') as f:
            f.write(s)
        return authority.PySourceAuthority(fp.path)

    def setUp(self):
        self.auth = self.loadPySourceString(samplePySource)

    def test_aRecords(self):
        """
        A records are loaded.
        """
        for dom, ip in [(b'example.com', '10.0.0.1'), (b'no-in.example.com', '10.0.0.2')]:
            [[rr], [], []] = self.successResultOf(self.auth.lookupAddress(dom))
            self.assertEqual(dns.Record_A(ip), rr.payload)

    def test_aaaaRecords(self):
        """
        AAAA records are loaded.
        """
        [[rr], [], []] = self.successResultOf(self.auth.lookupIPV6Address(b'example.com'))
        self.assertEqual(dns.Record_AAAA('2001:db8:10::1'), rr.payload)

    def test_mxRecords(self):
        """
        MX records are loaded.
        """
        [[rr], [], []] = self.successResultOf(self.auth.lookupMailExchange(b'not-fqdn.example.com'))
        self.assertEqual(dns.Record_MX(preference=10, name='mail.example.com'), rr.payload)

    def test_cnameRecords(self):
        """
        CNAME records are loaded.
        """
        [answers, [], []] = self.successResultOf(self.auth.lookupIPV6Address(b'www.example.com'))
        rr = answers[0]
        self.assertEqual(dns.Record_CNAME(name='example.com'), rr.payload)

    def test_PTR(self):
        """
        PTR records are loaded.
        """
        [answers, [], []] = self.successResultOf(self.auth.lookupPointer(b'2.0.0.10.in-addr.arpa'))
        rr = answers[0]
        self.assertEqual(dns.Record_PTR(name=b'no-in.example.com'), rr.payload)

    def test_badInputNoZone(self):
        """
        Input file has no zone variable
        """
        badPySource = 'nothing = []'
        self.assertRaises(ValueError, self.loadPySourceString, badPySource)
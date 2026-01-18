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
class AuthorityTests(unittest.TestCase):
    """
    Tests for the basic response record selection code in L{FileAuthority}
    (independent of its fileness).
    """

    def test_domainErrorForNameWithCommonSuffix(self):
        """
        L{FileAuthority} lookup methods errback with L{DomainError} if
        the requested C{name} shares a common suffix with its zone but
        is not actually a descendant of its zone, in terms of its
        sequence of DNS name labels. eg www.the-example.com has
        nothing to do with the zone example.com.
        """
        testDomain = test_domain_com
        testDomainName = b'nonexistent.prefix-' + testDomain.soa[0]
        f = self.failureResultOf(testDomain.lookupAddress(testDomainName))
        self.assertIsInstance(f.value, DomainError)

    def test_recordMissing(self):
        """
        If a L{FileAuthority} has a zone which includes an I{NS} record for a
        particular name and that authority is asked for another record for the
        same name which does not exist, the I{NS} record is not included in the
        authority section of the response.
        """
        authority = NoFileAuthority(soa=(soa_record.mname.name, soa_record), records={soa_record.mname.name: [soa_record, dns.Record_NS('1.2.3.4')]})
        answer, authority, additional = self.successResultOf(authority.lookupAddress(soa_record.mname.name))
        self.assertEqual(answer, [])
        self.assertEqual(authority, [dns.RRHeader(soa_record.mname.name, soa_record.TYPE, ttl=soa_record.expire, payload=soa_record, auth=True)])
        self.assertEqual(additional, [])

    def test_unknownTypeNXDOMAIN(self):
        """
        Requesting a record of unknown type where no records exist for the name
        in question results in L{DomainError}.
        """
        testDomain = test_domain_com
        testDomainName = b'nonexistent.prefix-' + testDomain.soa[0]
        unknownType = max(common.typeToMethod) + 1
        f = self.failureResultOf(testDomain.query(Query(name=testDomainName, type=unknownType)))
        self.assertIsInstance(f.value, DomainError)

    def test_unknownTypeMissing(self):
        """
        Requesting a record of unknown type where other records exist for the
        name in question results in an empty answer set.
        """
        unknownType = max(common.typeToMethod) + 1
        answer, authority, additional = self.successResultOf(my_domain_com.query(Query(name='my-domain.com', type=unknownType)))
        self.assertEqual(answer, [])

    def _referralTest(self, method):
        """
        Create an authority and make a request against it.  Then verify that the
        result is a referral, including no records in the answers or additional
        sections, but with an I{NS} record in the authority section.
        """
        subdomain = b'example.' + soa_record.mname.name
        nameserver = dns.Record_NS('1.2.3.4')
        authority = NoFileAuthority(soa=(soa_record.mname.name, soa_record), records={subdomain: [nameserver]})
        d = getattr(authority, method)(subdomain)
        answer, authority, additional = self.successResultOf(d)
        self.assertEqual(answer, [])
        self.assertEqual(authority, [dns.RRHeader(subdomain, dns.NS, ttl=soa_record.expire, payload=nameserver, auth=False)])
        self.assertEqual(additional, [])

    def test_referral(self):
        """
        When an I{NS} record is found for a child zone, it is included in the
        authority section of the response. It is marked as non-authoritative if
        the authority is not also authoritative for the child zone (RFC 2181,
        section 6.1).
        """
        self._referralTest('lookupAddress')

    def test_allRecordsReferral(self):
        """
        A referral is also generated for a request of type C{ALL_RECORDS}.
        """
        self._referralTest('lookupAllRecords')
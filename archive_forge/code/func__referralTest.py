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
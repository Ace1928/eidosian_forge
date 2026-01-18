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
def _answerCNAMETest(self, addresses):
    """
        Verify that a response to a CNAME query has certain records in the
        I{answer} section.

        @param addresses: See C{_additionalTest}
        """
    target = b'www.' + soa_record.mname.name
    d = self._lookupSomeRecords('lookupCanonicalName', soa_record, dns.Record_CNAME, target, addresses)
    answer, authority, additional = self.successResultOf(d)
    alias = dns.RRHeader(soa_record.mname.name, dns.CNAME, ttl=soa_record.expire, payload=dns.Record_CNAME(target), auth=True)
    self.assertRecordsMatch([dns.RRHeader(target, address.TYPE, ttl=soa_record.expire, payload=address, auth=True) for address in addresses] + [alias], answer)
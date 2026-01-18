import errno
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import defer
from twisted.internet.error import CannotListenError, ConnectionRefusedError
from twisted.internet.interfaces import IResolver
from twisted.internet.task import Clock
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.names import cache, client, dns, error, hosts
from twisted.names.common import ResolverBase
from twisted.names.error import DNSQueryTimeoutError
from twisted.names.test import test_util
from twisted.names.test.test_hosts import GoodTempPathMixin
from twisted.python import failure
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test import proto_helpers
from twisted.trial import unittest
class FakeResolver(ResolverBase):

    def _lookup(self, name, cls, qtype, timeout):
        """
        The getHostByNameTest does a different type of query that requires it
        return an A record from an ALL_RECORDS lookup, so we accommodate that
        here.
        """
        if name == b'getHostByNameTest':
            rr = dns.RRHeader(name=name, type=dns.A, cls=cls, ttl=60, payload=dns.Record_A(address='127.0.0.1', ttl=60))
        else:
            rr = dns.RRHeader(name=name, type=qtype, cls=cls, ttl=60)
        results = [rr]
        authority = []
        additional = []
        return defer.succeed((results, authority, additional))
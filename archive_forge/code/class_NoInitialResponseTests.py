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
class NoInitialResponseTests(unittest.TestCase):

    def test_noAnswer(self):
        """
        If a request returns a L{dns.NS} response, but we can't connect to the
        given server, the request fails with the error returned at connection.
        """

        def query(self, *args):
            return succeed(messages.pop(0))

        def queryProtocol(self, *args, **kwargs):
            return defer.fail(socket.gaierror("Couldn't connect"))
        resolver = Resolver(servers=[('0.0.0.0', 0)])
        resolver._query = query
        messages = []
        self.patch(dns.DNSDatagramProtocol, 'query', queryProtocol)
        records = [dns.RRHeader(name='fooba.com', type=dns.NS, cls=dns.IN, ttl=700, auth=False, payload=dns.Record_NS(name='ns.twistedmatrix.com', ttl=700))]
        m = dns.Message(id=999, answer=1, opCode=0, recDes=0, recAv=1, auth=1, rCode=0, trunc=0, maxSize=0)
        m.answers = records
        messages.append(m)
        return self.assertFailure(resolver.getHostByName('fooby.com'), socket.gaierror)
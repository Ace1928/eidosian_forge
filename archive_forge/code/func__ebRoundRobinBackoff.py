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
def _ebRoundRobinBackoff(self, failure, fakeProto):
    failure.trap(defer.TimeoutError)
    for t in (1, 3, 11, 45):
        tries = fakeProto.queries[:len(self.testServers)]
        del fakeProto.queries[:len(self.testServers)]
        tries.sort()
        expected = list(self.testServers)
        expected.sort()
        for (addr, query, timeout, id), expectedAddr in zip(tries, expected):
            self.assertEqual(addr, (expectedAddr, 53))
            self.assertEqual(timeout, t)
    self.assertFalse(fakeProto.queries)
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
class FilterAnswersTests(unittest.TestCase):
    """
    Test L{twisted.names.client.Resolver.filterAnswers}'s handling of various
    error conditions it might encounter.
    """

    def setUp(self):
        self.resolver = client.Resolver(servers=[('0.0.0.0', 0)])

    def test_truncatedMessage(self):
        """
        Test that a truncated message results in an equivalent request made via
        TCP.
        """
        m = dns.Message(trunc=True)
        m.addQuery(b'example.com')

        def queryTCP(queries):
            self.assertEqual(queries, m.queries)
            response = dns.Message()
            response.answers = ['answer']
            response.authority = ['authority']
            response.additional = ['additional']
            return defer.succeed(response)
        self.resolver.queryTCP = queryTCP
        d = self.resolver.filterAnswers(m)
        d.addCallback(self.assertEqual, (['answer'], ['authority'], ['additional']))
        return d

    def _rcodeTest(self, rcode, exc):
        m = dns.Message(rCode=rcode)
        err = self.resolver.filterAnswers(m)
        err.trap(exc)

    def test_formatError(self):
        """
        Test that a message with a result code of C{EFORMAT} results in a
        failure wrapped around L{DNSFormatError}.
        """
        return self._rcodeTest(dns.EFORMAT, error.DNSFormatError)

    def test_serverError(self):
        """
        Like L{test_formatError} but for C{ESERVER}/L{DNSServerError}.
        """
        return self._rcodeTest(dns.ESERVER, error.DNSServerError)

    def test_nameError(self):
        """
        Like L{test_formatError} but for C{ENAME}/L{DNSNameError}.
        """
        return self._rcodeTest(dns.ENAME, error.DNSNameError)

    def test_notImplementedError(self):
        """
        Like L{test_formatError} but for C{ENOTIMP}/L{DNSNotImplementedError}.
        """
        return self._rcodeTest(dns.ENOTIMP, error.DNSNotImplementedError)

    def test_refusedError(self):
        """
        Like L{test_formatError} but for C{EREFUSED}/L{DNSQueryRefusedError}.
        """
        return self._rcodeTest(dns.EREFUSED, error.DNSQueryRefusedError)

    def test_refusedErrorUnknown(self):
        """
        Like L{test_formatError} but for an unrecognized error code and
        L{DNSUnknownError}.
        """
        return self._rcodeTest(dns.EREFUSED + 1, error.DNSUnknownError)
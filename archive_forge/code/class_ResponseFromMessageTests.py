import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
class ResponseFromMessageTests(unittest.SynchronousTestCase):
    """
    Tests for L{dns._responseFromMessage}.
    """

    def test_responseFromMessageResponseType(self):
        """
        L{dns.Message._responseFromMessage} is a constructor function which
        generates a new I{answer} message from an existing L{dns.Message} like
        instance.
        """
        request = dns.Message()
        response = dns._responseFromMessage(responseConstructor=dns.Message, message=request)
        self.assertIsNot(request, response)

    def test_responseType(self):
        """
        L{dns._responseFromMessage} returns a new instance of C{cls}
        """

        class SuppliedClass:
            id = 1
            queries = []
        expectedClass = dns.Message
        self.assertIsInstance(dns._responseFromMessage(responseConstructor=expectedClass, message=SuppliedClass()), expectedClass)

    def test_responseId(self):
        """
        L{dns._responseFromMessage} copies the C{id} attribute of the original
        message.
        """
        self.assertEqual(1234, dns._responseFromMessage(responseConstructor=dns.Message, message=dns.Message(id=1234)).id)

    def test_responseAnswer(self):
        """
        L{dns._responseFromMessage} sets the C{answer} flag to L{True}
        """
        request = dns.Message()
        response = dns._responseFromMessage(responseConstructor=dns.Message, message=request)
        self.assertEqual((False, True), (request.answer, response.answer))

    def test_responseQueries(self):
        """
        L{dns._responseFromMessage} copies the C{queries} attribute of the
        original message.
        """
        request = dns.Message()
        expectedQueries = [object(), object(), object()]
        request.queries = expectedQueries[:]
        self.assertEqual(expectedQueries, dns._responseFromMessage(responseConstructor=dns.Message, message=request).queries)

    def test_responseKwargs(self):
        """
        L{dns._responseFromMessage} accepts other C{kwargs} which are assigned
        to the new message before it is returned.
        """
        self.assertEqual(123, dns._responseFromMessage(responseConstructor=dns.Message, message=dns.Message(), rCode=123).rCode)
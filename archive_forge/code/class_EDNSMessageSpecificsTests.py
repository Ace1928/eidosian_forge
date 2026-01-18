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
class EDNSMessageSpecificsTests(ConstructorTestsMixin, unittest.SynchronousTestCase):
    """
    Tests for L{dns._EDNSMessage}.

    These tests are for L{dns._EDNSMessage} APIs which are not shared with
    L{dns.Message}.
    """
    messageFactory = dns._EDNSMessage

    def test_ednsVersion(self):
        """
        L{dns._EDNSMessage.ednsVersion} defaults to C{0} and can be overridden
        in the constructor.
        """
        self._verifyConstructorArgument('ednsVersion', defaultVal=0, altVal=None)

    def test_dnssecOK(self):
        """
        L{dns._EDNSMessage.dnssecOK} defaults to C{False} and can be overridden
        in the constructor.
        """
        self._verifyConstructorFlag('dnssecOK', defaultVal=False)

    def test_authenticData(self):
        """
        L{dns._EDNSMessage.authenticData} defaults to C{False} and can be
        overridden in the constructor.
        """
        self._verifyConstructorFlag('authenticData', defaultVal=False)

    def test_checkingDisabled(self):
        """
        L{dns._EDNSMessage.checkingDisabled} defaults to C{False} and can be
        overridden in the constructor.
        """
        self._verifyConstructorFlag('checkingDisabled', defaultVal=False)

    def test_queriesOverride(self):
        """
        L{dns._EDNSMessage.queries} can be overridden in the constructor.
        """
        msg = self.messageFactory(queries=[dns.Query(b'example.com')])
        self.assertEqual(msg.queries, [dns.Query(b'example.com')])

    def test_answersOverride(self):
        """
        L{dns._EDNSMessage.answers} can be overridden in the constructor.
        """
        msg = self.messageFactory(answers=[dns.RRHeader(b'example.com', payload=dns.Record_A('1.2.3.4'))])
        self.assertEqual(msg.answers, [dns.RRHeader(b'example.com', payload=dns.Record_A('1.2.3.4'))])

    def test_authorityOverride(self):
        """
        L{dns._EDNSMessage.authority} can be overridden in the constructor.
        """
        msg = self.messageFactory(authority=[dns.RRHeader(b'example.com', type=dns.SOA, payload=dns.Record_SOA())])
        self.assertEqual(msg.authority, [dns.RRHeader(b'example.com', type=dns.SOA, payload=dns.Record_SOA())])

    def test_additionalOverride(self):
        """
        L{dns._EDNSMessage.authority} can be overridden in the constructor.
        """
        msg = self.messageFactory(additional=[dns.RRHeader(b'example.com', payload=dns.Record_A('1.2.3.4'))])
        self.assertEqual(msg.additional, [dns.RRHeader(b'example.com', payload=dns.Record_A('1.2.3.4'))])

    def test_reprDefaults(self):
        """
        L{dns._EDNSMessage.__repr__} omits field values and sections which are
        identical to their defaults. The id field value is always shown.
        """
        self.assertEqual('<_EDNSMessage id=0>', repr(self.messageFactory()))

    def test_reprFlagsIfSet(self):
        """
        L{dns._EDNSMessage.__repr__} displays flags if they are L{True}.
        """
        m = self.messageFactory(answer=True, auth=True, trunc=True, recDes=True, recAv=True, authenticData=True, checkingDisabled=True, dnssecOK=True)
        self.assertEqual('<_EDNSMessage id=0 flags=answer,auth,trunc,recDes,recAv,authenticData,checkingDisabled,dnssecOK>', repr(m))

    def test_reprNonDefautFields(self):
        """
        L{dns._EDNSMessage.__repr__} displays field values if they differ from
        their defaults.
        """
        m = self.messageFactory(id=10, opCode=20, rCode=30, maxSize=40, ednsVersion=50)
        self.assertEqual('<_EDNSMessage id=10 opCode=20 rCode=30 maxSize=40 ednsVersion=50>', repr(m))

    def test_reprNonDefaultSections(self):
        """
        L{dns.Message.__repr__} displays sections which differ from their
        defaults.
        """
        m = self.messageFactory()
        m.queries = [1, 2, 3]
        m.answers = [4, 5, 6]
        m.authority = [7, 8, 9]
        m.additional = [10, 11, 12]
        self.assertEqual('<_EDNSMessage id=0 queries=[1, 2, 3] answers=[4, 5, 6] authority=[7, 8, 9] additional=[10, 11, 12]>', repr(m))

    def test_fromStrCallsMessageFactory(self):
        """
        L{dns._EDNSMessage.fromString} calls L{dns._EDNSMessage._messageFactory}
        to create a new L{dns.Message} instance which is used to decode the
        supplied bytes.
        """

        class FakeMessageFactory:
            """
            Fake message factory.
            """

            def fromStr(self, *args, **kwargs):
                """
                Fake fromStr method which raises the arguments it was passed.

                @param args: positional arguments
                @param kwargs: keyword arguments
                """
                raise RaisedArgs(args, kwargs)
        m = dns._EDNSMessage()
        m._messageFactory = FakeMessageFactory
        dummyBytes = object()
        e = self.assertRaises(RaisedArgs, m.fromStr, dummyBytes)
        self.assertEqual(((dummyBytes,), {}), (e.args, e.kwargs))

    def test_fromStrCallsFromMessage(self):
        """
        L{dns._EDNSMessage.fromString} calls L{dns._EDNSMessage._fromMessage}
        with a L{dns.Message} instance
        """
        m = dns._EDNSMessage()

        class FakeMessageFactory:
            """
            Fake message factory.
            """

            def fromStr(self, bytes):
                """
                A noop fake version of fromStr

                @param bytes: the bytes to be decoded
                """
        fakeMessage = FakeMessageFactory()
        m._messageFactory = lambda: fakeMessage

        def fakeFromMessage(*args, **kwargs):
            raise RaisedArgs(args, kwargs)
        m._fromMessage = fakeFromMessage
        e = self.assertRaises(RaisedArgs, m.fromStr, b'')
        self.assertEqual(((fakeMessage,), {}), (e.args, e.kwargs))

    def test_toStrCallsToMessage(self):
        """
        L{dns._EDNSMessage.toStr} calls L{dns._EDNSMessage._toMessage}
        """
        m = dns._EDNSMessage()

        def fakeToMessage(*args, **kwargs):
            raise RaisedArgs(args, kwargs)
        m._toMessage = fakeToMessage
        e = self.assertRaises(RaisedArgs, m.toStr)
        self.assertEqual(((), {}), (e.args, e.kwargs))

    def test_toStrCallsToMessageToStr(self):
        """
        L{dns._EDNSMessage.toStr} calls C{toStr} on the message returned by
        L{dns._EDNSMessage._toMessage}.
        """
        m = dns._EDNSMessage()
        dummyBytes = object()

        class FakeMessage:
            """
            Fake Message
            """

            def toStr(self):
                """
                Fake toStr which returns dummyBytes.

                @return: dummyBytes
                """
                return dummyBytes

        def fakeToMessage(*args, **kwargs):
            return FakeMessage()
        m._toMessage = fakeToMessage
        self.assertEqual(dummyBytes, m.toStr())
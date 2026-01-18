import struct
import sys
from io import BytesIO
from typing import List, Optional, Type
from zope.interface.verify import verifyObject
from twisted.internet import protocol, task
from twisted.internet.interfaces import IProducer
from twisted.internet.protocol import connectionDone
from twisted.protocols import basic
from twisted.python.compat import iterbytes
from twisted.python.failure import Failure
from twisted.test import proto_helpers
from twisted.trial import unittest
class RecvdAttributeMixin:
    """
    Mixin defining tests for string receiving protocols with a C{recvd}
    attribute which should be settable by application code, to be combined with
    L{IntNTestCaseMixin} on a L{TestCase} subclass
    """

    def makeMessage(self, protocol, data):
        """
        Return C{data} prefixed with message length in C{protocol.structFormat}
        form.
        """
        return struct.pack(protocol.structFormat, len(data)) + data

    def test_recvdContainsRemainingData(self):
        """
        In stringReceived, recvd contains the remaining data that was passed to
        dataReceived that was not part of the current message.
        """
        result = []
        r = self.getProtocol()

        def stringReceived(receivedString):
            result.append(r.recvd)
        r.stringReceived = stringReceived
        completeMessage = struct.pack(r.structFormat, 5) + b'a' * 5
        incompleteMessage = struct.pack(r.structFormat, 5) + b'b' * 4
        r.dataReceived(completeMessage + incompleteMessage)
        self.assertEqual(result, [incompleteMessage])

    def test_recvdChanged(self):
        """
        In stringReceived, if recvd is changed, messages should be parsed from
        it rather than the input to dataReceived.
        """
        r = self.getProtocol()
        result = []
        payloadC = b'c' * 5
        messageC = self.makeMessage(r, payloadC)

        def stringReceived(receivedString):
            if not result:
                r.recvd = messageC
            result.append(receivedString)
        r.stringReceived = stringReceived
        payloadA = b'a' * 5
        payloadB = b'b' * 5
        messageA = self.makeMessage(r, payloadA)
        messageB = self.makeMessage(r, payloadB)
        r.dataReceived(messageA + messageB)
        self.assertEqual(result, [payloadA, payloadC])

    def test_switching(self):
        """
        Data already parsed by L{IntNStringReceiver.dataReceived} is not
        reparsed if C{stringReceived} consumes some of the
        L{IntNStringReceiver.recvd} buffer.
        """
        proto = self.getProtocol()
        mix = []
        SWITCH = b'\x00\x00\x00\x00'
        for s in self.strings:
            mix.append(self.makeMessage(proto, s))
            mix.append(SWITCH)
        result = []

        def stringReceived(receivedString):
            result.append(receivedString)
            proto.recvd = proto.recvd[len(SWITCH):]
        proto.stringReceived = stringReceived
        proto.dataReceived(b''.join(mix))
        proto.dataReceived(b'\x01')
        self.assertEqual(result, self.strings)
        self.assertEqual(proto.recvd, b'\x01')

    def test_recvdInLengthLimitExceeded(self):
        """
        The L{IntNStringReceiver.recvd} buffer contains all data not yet
        processed by L{IntNStringReceiver.dataReceived} if the
        C{lengthLimitExceeded} event occurs.
        """
        proto = self.getProtocol()
        DATA = b'too long'
        proto.MAX_LENGTH = len(DATA) - 1
        message = self.makeMessage(proto, DATA)
        result = []

        def lengthLimitExceeded(length):
            result.append(length)
            result.append(proto.recvd)
        proto.lengthLimitExceeded = lengthLimitExceeded
        proto.dataReceived(message)
        self.assertEqual(result[0], len(DATA))
        self.assertEqual(result[1], message)
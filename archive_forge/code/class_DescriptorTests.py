import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
class DescriptorTests(TestCase):
    """
    Tests for L{amp.Descriptor}, an argument type for passing a file descriptor
    over an AMP connection over a UNIX domain socket.
    """

    def setUp(self):
        self.fuzz = 3
        self.transport = UNIXStringTransport(descriptorFuzz=self.fuzz)
        self.protocol = amp.BinaryBoxProtocol(amp.BoxDispatcher(amp.CommandLocator()))
        self.protocol.makeConnection(self.transport)

    def test_fromStringProto(self):
        """
        L{Descriptor.fromStringProto} constructs a file descriptor value by
        extracting a previously received file descriptor corresponding to the
        wire value of the argument from the L{_DescriptorExchanger} state of the
        protocol passed to it.

        This is a whitebox test which involves direct L{_DescriptorExchanger}
        state inspection.
        """
        argument = amp.Descriptor()
        self.protocol.fileDescriptorReceived(5)
        self.protocol.fileDescriptorReceived(3)
        self.protocol.fileDescriptorReceived(1)
        self.assertEqual(5, argument.fromStringProto('0', self.protocol))
        self.assertEqual(3, argument.fromStringProto('1', self.protocol))
        self.assertEqual(1, argument.fromStringProto('2', self.protocol))
        self.assertEqual({}, self.protocol._descriptors)

    def test_toStringProto(self):
        """
        To send a file descriptor, L{Descriptor.toStringProto} uses the
        L{IUNIXTransport.sendFileDescriptor} implementation of the transport of
        the protocol passed to it to copy the file descriptor.  Each subsequent
        descriptor sent over a particular AMP connection is assigned the next
        integer value, starting from 0.  The base ten string representation of
        this value is the byte encoding of the argument.

        This is a whitebox test which involves direct L{_DescriptorExchanger}
        state inspection and mutation.
        """
        argument = amp.Descriptor()
        self.assertEqual(b'0', argument.toStringProto(2, self.protocol))
        self.assertEqual(('fileDescriptorReceived', 2 + self.fuzz), self.transport._queue.pop(0))
        self.assertEqual(b'1', argument.toStringProto(4, self.protocol))
        self.assertEqual(('fileDescriptorReceived', 4 + self.fuzz), self.transport._queue.pop(0))
        self.assertEqual(b'2', argument.toStringProto(6, self.protocol))
        self.assertEqual(('fileDescriptorReceived', 6 + self.fuzz), self.transport._queue.pop(0))
        self.assertEqual({}, self.protocol._descriptors)

    def test_roundTrip(self):
        """
        L{amp.Descriptor.fromBox} can interpret an L{amp.AmpBox} constructed by
        L{amp.Descriptor.toBox} to reconstruct a file descriptor value.
        """
        name = 'alpha'
        nameAsBytes = name.encode('ascii')
        strings = {}
        descriptor = 17
        sendObjects = {name: descriptor}
        argument = amp.Descriptor()
        argument.toBox(nameAsBytes, strings, sendObjects.copy(), self.protocol)
        receiver = amp.BinaryBoxProtocol(amp.BoxDispatcher(amp.CommandLocator()))
        for event in self.transport._queue:
            getattr(receiver, event[0])(*event[1:])
        receiveObjects = {}
        argument.fromBox(nameAsBytes, strings.copy(), receiveObjects, receiver)
        self.assertEqual(descriptor + self.fuzz, receiveObjects[name])
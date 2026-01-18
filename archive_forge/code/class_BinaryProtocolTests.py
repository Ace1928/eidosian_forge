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
class BinaryProtocolTests(TestCase):
    """
    Tests for L{amp.BinaryBoxProtocol}.

    @ivar _boxSender: After C{startReceivingBoxes} is called, the L{IBoxSender}
        which was passed to it.
    """

    def setUp(self):
        """
        Keep track of all boxes received by this test in its capacity as an
        L{IBoxReceiver} implementor.
        """
        self.boxes = []
        self.data = []

    def startReceivingBoxes(self, sender):
        """
        Implement L{IBoxReceiver.startReceivingBoxes} to just remember the
        value passed in.
        """
        self._boxSender = sender

    def ampBoxReceived(self, box):
        """
        A box was received by the protocol.
        """
        self.boxes.append(box)
    stopReason = None

    def stopReceivingBoxes(self, reason):
        """
        Record the reason that we stopped receiving boxes.
        """
        self.stopReason = reason

    def getPeer(self):
        return 'no peer'

    def getHost(self):
        return 'no host'

    def write(self, data):
        self.assertIsInstance(data, bytes)
        self.data.append(data)

    def test_startReceivingBoxes(self):
        """
        When L{amp.BinaryBoxProtocol} is connected to a transport, it calls
        C{startReceivingBoxes} on its L{IBoxReceiver} with itself as the
        L{IBoxSender} parameter.
        """
        protocol = amp.BinaryBoxProtocol(self)
        protocol.makeConnection(None)
        self.assertIs(self._boxSender, protocol)

    def test_sendBoxInStartReceivingBoxes(self):
        """
        The L{IBoxReceiver} which is started when L{amp.BinaryBoxProtocol} is
        connected to a transport can call C{sendBox} on the L{IBoxSender}
        passed to it before C{startReceivingBoxes} returns and have that box
        sent.
        """

        class SynchronouslySendingReceiver:

            def startReceivingBoxes(self, sender):
                sender.sendBox(amp.Box({b'foo': b'bar'}))
        transport = StringTransport()
        protocol = amp.BinaryBoxProtocol(SynchronouslySendingReceiver())
        protocol.makeConnection(transport)
        self.assertEqual(transport.value(), b'\x00\x03foo\x00\x03bar\x00\x00')

    def test_receiveBoxStateMachine(self):
        """
        When a binary box protocol receives:
            * a key
            * a value
            * an empty string
        it should emit a box and send it to its boxReceiver.
        """
        a = amp.BinaryBoxProtocol(self)
        a.stringReceived(b'hello')
        a.stringReceived(b'world')
        a.stringReceived(b'')
        self.assertEqual(self.boxes, [amp.AmpBox(hello=b'world')])

    def test_firstBoxFirstKeyExcessiveLength(self):
        """
        L{amp.BinaryBoxProtocol} drops its connection if the length prefix for
        the first a key it receives is larger than 255.
        """
        transport = StringTransport()
        protocol = amp.BinaryBoxProtocol(self)
        protocol.makeConnection(transport)
        protocol.dataReceived(b'\x01\x00')
        self.assertTrue(transport.disconnecting)

    def test_firstBoxSubsequentKeyExcessiveLength(self):
        """
        L{amp.BinaryBoxProtocol} drops its connection if the length prefix for
        a subsequent key in the first box it receives is larger than 255.
        """
        transport = StringTransport()
        protocol = amp.BinaryBoxProtocol(self)
        protocol.makeConnection(transport)
        protocol.dataReceived(b'\x00\x01k\x00\x01v')
        self.assertFalse(transport.disconnecting)
        protocol.dataReceived(b'\x01\x00')
        self.assertTrue(transport.disconnecting)

    def test_subsequentBoxFirstKeyExcessiveLength(self):
        """
        L{amp.BinaryBoxProtocol} drops its connection if the length prefix for
        the first key in a subsequent box it receives is larger than 255.
        """
        transport = StringTransport()
        protocol = amp.BinaryBoxProtocol(self)
        protocol.makeConnection(transport)
        protocol.dataReceived(b'\x00\x01k\x00\x01v\x00\x00')
        self.assertFalse(transport.disconnecting)
        protocol.dataReceived(b'\x01\x00')
        self.assertTrue(transport.disconnecting)

    def test_excessiveKeyFailure(self):
        """
        If L{amp.BinaryBoxProtocol} disconnects because it received a key
        length prefix which was too large, the L{IBoxReceiver}'s
        C{stopReceivingBoxes} method is called with a L{TooLong} failure.
        """
        protocol = amp.BinaryBoxProtocol(self)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'\x01\x00')
        protocol.connectionLost(Failure(error.ConnectionDone('simulated connection done')))
        self.stopReason.trap(amp.TooLong)
        self.assertTrue(self.stopReason.value.isKey)
        self.assertFalse(self.stopReason.value.isLocal)
        self.assertIsNone(self.stopReason.value.value)
        self.assertIsNone(self.stopReason.value.keyName)

    def test_unhandledErrorWithTransport(self):
        """
        L{amp.BinaryBoxProtocol.unhandledError} logs the failure passed to it
        and disconnects its transport.
        """
        transport = StringTransport()
        protocol = amp.BinaryBoxProtocol(self)
        protocol.makeConnection(transport)
        protocol.unhandledError(Failure(RuntimeError('Fake error')))
        self.assertEqual(1, len(self.flushLoggedErrors(RuntimeError)))
        self.assertTrue(transport.disconnecting)

    def test_unhandledErrorWithoutTransport(self):
        """
        L{amp.BinaryBoxProtocol.unhandledError} completes without error when
        there is no associated transport.
        """
        protocol = amp.BinaryBoxProtocol(self)
        protocol.makeConnection(StringTransport())
        protocol.connectionLost(Failure(Exception('Simulated')))
        protocol.unhandledError(Failure(RuntimeError('Fake error')))
        self.assertEqual(1, len(self.flushLoggedErrors(RuntimeError)))

    def test_receiveBoxData(self):
        """
        When a binary box protocol receives the serialized form of an AMP box,
        it should emit a similar box to its boxReceiver.
        """
        a = amp.BinaryBoxProtocol(self)
        a.dataReceived(amp.Box({b'testKey': b'valueTest', b'anotherKey': b'anotherValue'}).serialize())
        self.assertEqual(self.boxes, [amp.Box({b'testKey': b'valueTest', b'anotherKey': b'anotherValue'})])

    def test_receiveLongerBoxData(self):
        """
        An L{amp.BinaryBoxProtocol} can receive serialized AMP boxes with
        values of up to (2 ** 16 - 1) bytes.
        """
        length = 2 ** 16 - 1
        value = b'x' * length
        transport = StringTransport()
        protocol = amp.BinaryBoxProtocol(self)
        protocol.makeConnection(transport)
        protocol.dataReceived(amp.Box({'k': value}).serialize())
        self.assertEqual(self.boxes, [amp.Box({'k': value})])
        self.assertFalse(transport.disconnecting)

    def test_sendBox(self):
        """
        When a binary box protocol sends a box, it should emit the serialized
        bytes of that box to its transport.
        """
        a = amp.BinaryBoxProtocol(self)
        a.makeConnection(self)
        aBox = amp.Box({b'testKey': b'valueTest', b'someData': b'hello'})
        a.makeConnection(self)
        a.sendBox(aBox)
        self.assertEqual(b''.join(self.data), aBox.serialize())

    def test_connectionLostStopSendingBoxes(self):
        """
        When a binary box protocol loses its connection, it should notify its
        box receiver that it has stopped receiving boxes.
        """
        a = amp.BinaryBoxProtocol(self)
        a.makeConnection(self)
        connectionFailure = Failure(RuntimeError())
        a.connectionLost(connectionFailure)
        self.assertIs(self.stopReason, connectionFailure)

    def test_protocolSwitch(self):
        """
        L{BinaryBoxProtocol} has the capacity to switch to a different protocol
        on a box boundary.  When a protocol is in the process of switching, it
        cannot receive traffic.
        """
        otherProto = TestProto(None, b'outgoing data')
        test = self

        class SwitchyReceiver:
            switched = False

            def startReceivingBoxes(self, sender):
                pass

            def ampBoxReceived(self, box):
                test.assertFalse(self.switched, 'Should only receive one box!')
                self.switched = True
                a._lockForSwitch()
                a._switchTo(otherProto)
        a = amp.BinaryBoxProtocol(SwitchyReceiver())
        anyOldBox = amp.Box({b'include': b'lots', b'of': b'data'})
        a.makeConnection(self)
        moreThanOneBox = anyOldBox.serialize() + b'\x00\x00Hello, world!'
        a.dataReceived(moreThanOneBox)
        self.assertIs(otherProto.transport, self)
        self.assertEqual(b''.join(otherProto.data), b'\x00\x00Hello, world!')
        self.assertEqual(self.data, [b'outgoing data'])
        a.dataReceived(b'more data')
        self.assertEqual(b''.join(otherProto.data), b'\x00\x00Hello, world!more data')
        self.assertRaises(amp.ProtocolSwitched, a.sendBox, anyOldBox)

    def test_protocolSwitchEmptyBuffer(self):
        """
        After switching to a different protocol, if no extra bytes beyond
        the switch box were delivered, an empty string is not passed to the
        switched protocol's C{dataReceived} method.
        """
        a = amp.BinaryBoxProtocol(self)
        a.makeConnection(self)
        otherProto = TestProto(None, b'')
        a._switchTo(otherProto)
        self.assertEqual(otherProto.data, [])

    def test_protocolSwitchInvalidStates(self):
        """
        In order to make sure the protocol never gets any invalid data sent
        into the middle of a box, it must be locked for switching before it is
        switched.  It can only be unlocked if the switch failed, and attempting
        to send a box while it is locked should raise an exception.
        """
        a = amp.BinaryBoxProtocol(self)
        a.makeConnection(self)
        sampleBox = amp.Box({b'some': b'data'})
        a._lockForSwitch()
        self.assertRaises(amp.ProtocolSwitched, a.sendBox, sampleBox)
        a._unlockFromSwitch()
        a.sendBox(sampleBox)
        self.assertEqual(b''.join(self.data), sampleBox.serialize())
        a._lockForSwitch()
        otherProto = TestProto(None, b'outgoing data')
        a._switchTo(otherProto)
        self.assertRaises(amp.ProtocolSwitched, a._unlockFromSwitch)

    def test_protocolSwitchLoseConnection(self):
        """
        When the protocol is switched, it should notify its nested protocol of
        disconnection.
        """

        class Loser(protocol.Protocol):
            reason = None

            def connectionLost(self, reason):
                self.reason = reason
        connectionLoser = Loser()
        a = amp.BinaryBoxProtocol(self)
        a.makeConnection(self)
        a._lockForSwitch()
        a._switchTo(connectionLoser)
        connectionFailure = Failure(RuntimeError())
        a.connectionLost(connectionFailure)
        self.assertEqual(connectionLoser.reason, connectionFailure)

    def test_protocolSwitchLoseClientConnection(self):
        """
        When the protocol is switched, it should notify its nested client
        protocol factory of disconnection.
        """

        class ClientLoser:
            reason = None

            def clientConnectionLost(self, connector, reason):
                self.reason = reason
        a = amp.BinaryBoxProtocol(self)
        connectionLoser = protocol.Protocol()
        clientLoser = ClientLoser()
        a.makeConnection(self)
        a._lockForSwitch()
        a._switchTo(connectionLoser, clientLoser)
        connectionFailure = Failure(RuntimeError())
        a.connectionLost(connectionFailure)
        self.assertEqual(clientLoser.reason, connectionFailure)
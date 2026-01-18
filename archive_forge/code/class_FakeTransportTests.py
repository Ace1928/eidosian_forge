from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
class FakeTransportTests(TestCase):
    """
    Tests for L{FakeTransport}.
    """

    def test_connectionSerial(self) -> None:
        """
        Each L{FakeTransport} receives a serial number that uniquely identifies
        it.
        """
        a = FakeTransport(object(), True)
        b = FakeTransport(object(), False)
        self.assertIsInstance(a.serial, int)
        self.assertIsInstance(b.serial, int)
        self.assertNotEqual(a.serial, b.serial)

    def test_writeSequence(self) -> None:
        """
        L{FakeTransport.writeSequence} will write a sequence of L{bytes} to the
        transport.
        """
        a = FakeTransport(object(), False)
        a.write(b'a')
        a.writeSequence([b'b', b'c', b'd'])
        self.assertEqual(b''.join(a.stream), b'abcd')

    def test_writeAfterClose(self) -> None:
        """
        L{FakeTransport.write} will accept writes after transport was closed,
        but the data will be silently discarded.
        """
        a = FakeTransport(object(), False)
        a.write(b'before')
        a.loseConnection()
        a.write(b'after')
        self.assertEqual(b''.join(a.stream), b'before')
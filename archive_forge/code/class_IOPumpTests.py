from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
class IOPumpTests(TestCase):
    """
    Tests for L{IOPump}.
    """

    def _testStreamingProducer(self, mode: Literal['server', 'client']) -> None:
        """
        Connect a couple protocol/transport pairs to an L{IOPump} and then pump
        it.  Verify that a streaming producer registered with one of the
        transports does not receive invalid L{IPushProducer} method calls and
        ends in the right state.

        @param mode: C{u"server"} to test a producer registered with the
            server transport.  C{u"client"} to test a producer registered with
            the client transport.
        """
        serverProto = Protocol()
        serverTransport = FakeTransport(serverProto, isServer=True)
        clientProto = Protocol()
        clientTransport = FakeTransport(clientProto, isServer=False)
        pump = connect(serverProto, serverTransport, clientProto, clientTransport, greet=False)
        producer = StrictPushProducer()
        victim = {'server': serverTransport, 'client': clientTransport}[mode]
        victim.registerProducer(producer, streaming=True)
        pump.pump()
        self.assertEqual('running', producer._state)

    def test_serverStreamingProducer(self) -> None:
        """
        L{IOPump.pump} does not call C{resumeProducing} on a L{IPushProducer}
        (stream producer) registered with the server transport.
        """
        self._testStreamingProducer(mode='server')

    def test_clientStreamingProducer(self) -> None:
        """
        L{IOPump.pump} does not call C{resumeProducing} on a L{IPushProducer}
        (stream producer) registered with the client transport.
        """
        self._testStreamingProducer(mode='client')

    def test_timeAdvances(self) -> None:
        """
        L{IOPump.pump} advances time in the given L{Clock}.
        """
        time_passed = []
        clock = Clock()
        _, _, pump = connectedServerAndClient(Protocol, Protocol, clock=clock)
        clock.callLater(0, lambda: time_passed.append(True))
        self.assertFalse(time_passed)
        pump.pump()
        self.assertTrue(time_passed)
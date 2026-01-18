from __future__ import annotations
import datetime
from operator import attrgetter
from typing import Callable, Iterable, TypedDict
from zope.interface import implementer
from constantly import NamedConstant
from typing_extensions import Literal, Protocol
from twisted.positioning import base, ipositioning, nmea
from twisted.positioning.base import Angles
from twisted.positioning.test.receiver import MockPositioningReceiver
from twisted.trial.unittest import TestCase
class NMEAReceiverSetup:
    """
    A mixin for tests that need an NMEA receiver (and a protocol attached to
    it).

    @ivar receiver: An NMEA receiver that remembers the last sentence.
    @type receiver: L{NMEATestReceiver}
    @ivar protocol: An NMEA protocol attached to the receiver.
    @type protocol: L{twisted.positioning.nmea.NMEAProtocol}
    """

    def setUp(self) -> None:
        """
        Sets up an NMEA receiver.
        """
        self.receiver = NMEATestReceiver()
        self.protocol = nmea.NMEAProtocol(self.receiver)
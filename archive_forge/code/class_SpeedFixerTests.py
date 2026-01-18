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
class SpeedFixerTests(FixerTestMixin, TestCase):
    """
    Tests that NMEA representations of speeds are correctly converted.
    """

    def test_speedInKnots(self) -> None:
        """
        Speeds reported in knots correctly get converted to meters per
        second.
        """
        key, value = ('speedInKnots', '10')
        speed = base.Speed(float(value) * base.MPS_PER_KNOT)
        self._fixerTest({key: value}, _State(speed=speed))
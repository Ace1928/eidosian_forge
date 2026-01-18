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
class AltitudeFixerTests(FixerTestMixin, TestCase):
    """
    Tests that NMEA representations of altitudes are correctly converted.
    """

    def test_fixAltitude(self) -> None:
        """
        The NMEA representation of an altitude (above mean sea level)
        is correctly converted.
        """
        key, value = ('altitude', '545.4')
        altitude = base.Altitude(float(value))
        self._fixerTest({key: value}, _State(altitude=altitude))

    def test_heightOfGeoidAboveWGS84(self) -> None:
        """
        The NMEA representation of an altitude of the geoid (above the
        WGS84 reference level) is correctly converted.
        """
        key, value = ('heightOfGeoidAboveWGS84', '46.9')
        altitude = base.Altitude(float(value))
        self._fixerTest({key: value}, _State(heightOfGeoidAboveWGS84=altitude))
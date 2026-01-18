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
class CoordinateFixerTests(FixerTestMixin, TestCase):
    """
    Tests turning NMEA coordinate notations into something more pleasant.
    """

    def test_north(self) -> None:
        """
        NMEA coordinate representations in the northern hemisphere
        convert correctly.
        """
        sentenceData = {'latitudeFloat': '1030.000', 'latitudeHemisphere': 'N'}
        state: _State = {'latitude': base.Coordinate(10.5, Angles.LATITUDE)}
        self._fixerTest(sentenceData, state)

    def test_south(self) -> None:
        """
        NMEA coordinate representations in the southern hemisphere
        convert correctly.
        """
        sentenceData = {'latitudeFloat': '1030.000', 'latitudeHemisphere': 'S'}
        state: _State = {'latitude': base.Coordinate(-10.5, Angles.LATITUDE)}
        self._fixerTest(sentenceData, state)

    def test_east(self) -> None:
        """
        NMEA coordinate representations in the eastern hemisphere
        convert correctly.
        """
        sentenceData = {'longitudeFloat': '1030.000', 'longitudeHemisphere': 'E'}
        state: _State = {'longitude': base.Coordinate(10.5, Angles.LONGITUDE)}
        self._fixerTest(sentenceData, state)

    def test_west(self) -> None:
        """
        NMEA coordinate representations in the western hemisphere
        convert correctly.
        """
        sentenceData = {'longitudeFloat': '1030.000', 'longitudeHemisphere': 'W'}
        state: _State = {'longitude': base.Coordinate(-10.5, Angles.LONGITUDE)}
        self._fixerTest(sentenceData, state)

    def test_badHemisphere(self) -> None:
        """
        NMEA coordinate representations for nonexistent hemispheres
        raise C{ValueError} when you attempt to parse them.
        """
        sentenceData = {'longitudeHemisphere': 'Q'}
        self._fixerTest(sentenceData, exceptionClass=ValueError)

    def test_badHemisphereSign(self) -> None:
        """
        NMEA coordinate repesentation parsing fails predictably
        when you pass nonexistent coordinate types (not latitude or
        longitude).
        """
        getSign = lambda: self.adapter._getHemisphereSign('BOGUS_VALUE')
        self.assertRaises(ValueError, getSign)
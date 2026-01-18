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
class ValidFixTests(FixerTestMixin, TestCase):
    """
    Tests that data reported from a valid fix is used.
    """

    def test_GGA(self) -> None:
        """
        GGA data with a valid fix is used.
        """
        sentenceData = {'type': 'GPGGA', 'altitude': '545.4', 'fixQuality': nmea.GPGGAFixQualities.GPS_FIX}
        expectedState: _State = {'altitude': base.Altitude(545.4)}
        self._fixerTest(sentenceData, expectedState)

    def test_GLL(self) -> None:
        """
        GLL data with a valid data mode is used.
        """
        sentenceData = {'type': 'GPGLL', 'altitude': '545.4', 'dataMode': nmea.GPGLLGPRMCFixQualities.ACTIVE}
        expectedState: _State = {'altitude': base.Altitude(545.4)}
        self._fixerTest(sentenceData, expectedState)
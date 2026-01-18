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
class PositionErrorFixerTests(FixerTestMixin, TestCase):
    """
    Position errors in NMEA are passed as dilutions of precision (DOP). This
    is a measure relative to some specified value of the GPS device as its
    "reference" precision. Unfortunately, there are very few ways of figuring
    this out from just the device (sans manual).

    There are two basic DOP values: vertical and horizontal. HDOP tells you
    how precise your location is on the face of the earth (pretending it's
    flat, at least locally). VDOP tells you how precise your altitude is
    known. PDOP (position DOP) is a dependent value defined as the Euclidean
    norm of those two, and gives you a more generic "goodness of fix" value.
    """

    def test_simple(self) -> None:
        self._fixerTest({'horizontalDilutionOfPrecision': '11'}, {'positionError': base.PositionError(hdop=11.0)})

    def test_mixing(self) -> None:
        pdop, hdop, vdop = ('1', '1', '1')
        positionError = base.PositionError(pdop=float(pdop), hdop=float(hdop), vdop=float(vdop))
        sentenceData = {'positionDilutionOfPrecision': pdop, 'horizontalDilutionOfPrecision': hdop, 'verticalDilutionOfPrecision': vdop}
        self._fixerTest(sentenceData, {'positionError': positionError})
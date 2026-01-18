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
class VariationFixerTests(FixerTestMixin, TestCase):
    """
    Tests if the absolute values of magnetic variations on the heading
    and their sign get combined correctly, and if that value gets
    combined with a heading correctly.
    """

    def test_west(self) -> None:
        """
        Tests westward (negative) magnetic variation.
        """
        variation, direction = ('1.34', 'W')
        heading = base.Heading.fromFloats(variationValue=-1 * float(variation))
        sentenceData = {'magneticVariation': variation, 'magneticVariationDirection': direction}
        self._fixerTest(sentenceData, {'heading': heading})

    def test_east(self) -> None:
        """
        Tests eastward (positive) magnetic variation.
        """
        variation, direction = ('1.34', 'E')
        heading = base.Heading.fromFloats(variationValue=float(variation))
        sentenceData = {'magneticVariation': variation, 'magneticVariationDirection': direction}
        self._fixerTest(sentenceData, {'heading': heading})

    def test_withHeading(self) -> None:
        """
        Variation values get combined with headings correctly.
        """
        trueHeading, variation, direction = ('123.12', '1.34', 'E')
        sentenceData = {'trueHeading': trueHeading, 'magneticVariation': variation, 'magneticVariationDirection': direction}
        heading = base.Heading.fromFloats(float(trueHeading), variationValue=float(variation))
        self._fixerTest(sentenceData, {'heading': heading})
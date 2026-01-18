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
class DatestampFixerTests(FixerTestMixin, TestCase):

    def test_defaultYearThreshold(self) -> None:
        """
        The default year threshold is 1980.
        """
        self.assertEqual(self.adapter.yearThreshold, 1980)

    def test_beforeThreshold(self) -> None:
        """
        Dates before the threshold are interpreted as being in the century
        after the threshold. (Since the threshold is the earliest possible
        date.)
        """
        datestring, date = ('010115', datetime.date(2015, 1, 1))
        self._fixerTest({'datestamp': datestring}, {'_date': date})

    def test_afterThreshold(self) -> None:
        """
        Dates after the threshold are interpreted as being in the same century
        as the threshold.
        """
        datestring, date = ('010195', datetime.date(1995, 1, 1))
        self._fixerTest({'datestamp': datestring}, {'_date': date})

    def test_invalidMonth(self) -> None:
        """
        A datestring with an invalid month (> 12) raises C{ValueError}.
        """
        self._fixerTest({'datestamp': '011301'}, exceptionClass=ValueError)

    def test_invalidDay(self) -> None:
        """
        A datestring with an invalid day (more days than there are in that
        month) raises C{ValueError}.
        """
        self._fixerTest({'datestamp': '320101'}, exceptionClass=ValueError)
        self._fixerTest({'datestamp': '300201'}, exceptionClass=ValueError)
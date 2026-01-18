from __future__ import annotations
import calendar
import logging
import os
import sys
import time
import warnings
from io import IOBase, StringIO
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import Protocol
from twisted.logger import (
from twisted.logger.test.test_stdlib import handlerAndBytesIO
from twisted.python import failure, log
from twisted.python.log import LogPublisher
from twisted.trial import unittest
def _getTimezoneOffsetTest(self, tzname: str, daylightOffset: int, standardOffset: int) -> None:
    """
        Verify that L{getTimezoneOffset} produces the expected offset for a
        certain timezone both when daylight saving time is in effect and when
        it is not.

        @param tzname: The name of a timezone to exercise.
        @type tzname: L{bytes}

        @param daylightOffset: The number of seconds west of UTC the timezone
            should be when daylight saving time is in effect.
        @type daylightOffset: L{int}

        @param standardOffset: The number of seconds west of UTC the timezone
            should be when daylight saving time is not in effect.
        @type standardOffset: L{int}
        """
    if getattr(time, 'tzset', None) is None:
        raise unittest.SkipTest('Platform cannot change timezone, cannot verify correct offsets in well-known timezones.')
    originalTimezone = os.environ.get('TZ', None)
    try:
        os.environ['TZ'] = tzname
        time.tzset()
        localStandardTuple = (2007, 1, 31, 0, 0, 0, 2, 31, 0)
        standard = time.mktime(localStandardTuple)
        localDaylightTuple = (2006, 6, 30, 0, 0, 0, 4, 181, 1)
        try:
            daylight = time.mktime(localDaylightTuple)
        except OverflowError:
            if daylightOffset == standardOffset:
                daylight = standard
            else:
                raise
        self.assertEqual((self.flo.getTimezoneOffset(daylight), self.flo.getTimezoneOffset(standard)), (daylightOffset, standardOffset))
    finally:
        if originalTimezone is None:
            del os.environ['TZ']
        else:
            os.environ['TZ'] = originalTimezone
        time.tzset()
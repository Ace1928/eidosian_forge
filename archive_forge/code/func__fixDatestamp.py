import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _fixDatestamp(self):
    """
        Turns an NMEA datestamp format into a C{datetime.date} object.

        @raise ValueError: When the day or month value was invalid, e.g. 32nd
            day, or 13th month, or 0th day or month.
        """
    date = self.currentSentence.datestamp
    day, month, year = map(int, [date[0:2], date[2:4], date[4:6]])
    year += self.yearThreshold - self.yearThreshold % 100
    if year < self.yearThreshold:
        year += 100
    self._sentenceData['_date'] = datetime.date(year, month, day)
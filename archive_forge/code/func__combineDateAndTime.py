import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _combineDateAndTime(self):
    """
        Combines a C{datetime.date} object and a C{datetime.time} object,
        collected from one or more NMEA sentences, into a single
        C{datetime.datetime} object suitable for sending to the
        L{IPositioningReceiver}.
        """
    if not any((k in self._sentenceData for k in ['_date', '_time'])):
        return
    date, time = (self._sentenceData.get(key) or self._state.get(key) for key in ('_date', '_time'))
    if date is None or time is None:
        return
    dt = datetime.datetime.combine(date, time)
    self._sentenceData['time'] = dt
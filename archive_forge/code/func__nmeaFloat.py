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
def _nmeaFloat(degrees: int, minutes: float) -> str:
    """
    Builds an NMEA float representation for a given angle in degrees and
    decimal minutes.

    @param degrees: The integer degrees for this angle.
    @type degrees: C{int}
    @param minutes: The decimal minutes value for this angle.
    @type minutes: C{float}
    @return: The NMEA float representation for this angle.
    @rtype: C{str}
    """
    return '%i%0.3f' % (degrees, minutes)
from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
def _setDOP(self, dopType, value):
    """
        Sets a particular dilution of position value.

        @param dopType: The type of dilution of position to set. One of
            ('pdop', 'hdop', 'vdop').
        @type dopType: C{str}

        @param value: The value to set the dilution of position type to.
        @type value: C{float}

        If this position error tests dilution of precision invariants,
        it will be checked. If the invariant is not satisfied, the
        assignment will be undone and C{ValueError} is raised.
        """
    attributeName = '_' + dopType
    oldValue = getattr(self, attributeName)
    setattr(self, attributeName, float(value))
    try:
        self._testDilutionOfPositionInvariant()
    except ValueError:
        setattr(self, attributeName, oldValue)
        raise
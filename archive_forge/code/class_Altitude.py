from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
class Altitude(FancyEqMixin):
    """
    An altitude.

    @ivar inMeters: The altitude represented by this object, in meters. This
        attribute is read-only.
    @type inMeters: C{float}

    @ivar inFeet: As above, but expressed in feet.
    @type inFeet: C{float}
    """
    compareAttributes = ('inMeters',)

    def __init__(self, altitude):
        """
        Initializes an altitude.

        @param altitude: The altitude in meters.
        @type altitude: C{float}
        """
        self._altitude = altitude

    @property
    def inFeet(self):
        """
        Gets the altitude this object represents, in feet.

        @return: The altitude, expressed in feet.
        @rtype: C{float}
        """
        return self._altitude / METERS_PER_FOOT

    @property
    def inMeters(self):
        """
        Returns the altitude this object represents, in meters.

        @return: The altitude, expressed in feet.
        @rtype: C{float}
        """
        return self._altitude

    def __float__(self):
        """
        Returns the altitude represented by this object expressed in meters.

        @return: The altitude represented by this object, expressed in meters.
        @rtype: C{float}
        """
        return self._altitude

    def __repr__(self) -> str:
        """
        Returns a string representation of this altitude.

        @return: The string representation.
        @rtype: C{str}
        """
        return f'<Altitude ({self._altitude} m)>'
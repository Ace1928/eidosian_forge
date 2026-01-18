from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
class _BaseSpeed(FancyEqMixin):
    """
    An object representing the abstract concept of the speed (rate of
    movement) of a mobile object.

    This primarily has behavior for converting between units and comparison.
    """
    compareAttributes = ('inMetersPerSecond',)

    def __init__(self, speed):
        """
        Initializes a speed.

        @param speed: The speed that this object represents, expressed in
            meters per second.
        @type speed: C{float}

        @raises ValueError: Raised if value was invalid for this particular
            kind of speed. Only happens in subclasses.
        """
        self._speed = speed

    @property
    def inMetersPerSecond(self):
        """
        The speed that this object represents, expressed in meters per second.
        This attribute is immutable.

        @return: The speed this object represents, in meters per second.
        @rtype: C{float}
        """
        return self._speed

    @property
    def inKnots(self):
        """
        Returns the speed represented by this object, expressed in knots. This
        attribute is immutable.

        @return: The speed this object represents, in knots.
        @rtype: C{float}
        """
        return self._speed / MPS_PER_KNOT

    def __float__(self):
        """
        Returns the speed represented by this object expressed in meters per
        second.

        @return: The speed represented by this object, expressed in meters per
            second.
        @rtype: C{float}
        """
        return self._speed

    def __repr__(self) -> str:
        """
        Returns a string representation of this speed object.

        @return: The string representation.
        @rtype: C{str}
        """
        speedValue = round(self.inMetersPerSecond, 2)
        return f'<{self.__class__.__name__} ({speedValue} m/s)>'
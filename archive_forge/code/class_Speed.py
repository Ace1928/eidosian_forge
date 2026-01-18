from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
class Speed(_BaseSpeed):
    """
    The speed (rate of movement) of a mobile object.
    """

    def __init__(self, speed):
        """
        Initializes a L{Speed} object.

        @param speed: The speed that this object represents, expressed in
            meters per second.
        @type speed: C{float}

        @raises ValueError: Raised if C{speed} is negative.
        """
        if speed < 0:
            raise ValueError(f'negative speed: {speed!r}')
        _BaseSpeed.__init__(self, speed)
from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
def _getDOP(self, dopType):
    """
        Gets a particular dilution of position value.

        @param dopType: The type of dilution of position to get. One of
            ('pdop', 'hdop', 'vdop').
        @type dopType: C{str}
        @return: The DOP if it is known, L{None} otherwise.
        @rtype: C{float} or L{None}
        """
    for dopExpression in self._DOP_EXPRESSIONS[dopType]:
        try:
            return dopExpression(self)
        except TypeError:
            continue
from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
@property
def _angleValueRepr(self):
    """
        Returns a string representation of the angular value of this angle.

        This is a helper function for the actual C{__repr__}.

        @return: The string representation.
        @rtype: C{str}
        """
    if self.inDecimalDegrees is not None:
        return '%s degrees' % round(self.inDecimalDegrees, 2)
    else:
        return 'unknown value'
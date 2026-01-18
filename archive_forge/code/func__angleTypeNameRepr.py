from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
@property
def _angleTypeNameRepr(self):
    """
        Returns a string representation of the type of this angle.

        This is a helper function for the actual C{__repr__}.

        @return: The string representation.
        @rtype: C{str}
        """
    try:
        return self._ANGLE_TYPE_NAMES[self.angleType]
    except KeyError:
        return 'Angle of unknown type'
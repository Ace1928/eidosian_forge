import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _getHemisphereSign(self, coordinateType):
    """
        Returns the hemisphere sign for a given coordinate type.

        @param coordinateType: The coordinate type to find the hemisphere for.
        @type coordinateType: L{Angles.LATITUDE}, L{Angles.LONGITUDE} or
            L{Angles.VARIATION}.
        @return: The sign of that hemisphere (-1 or 1).
        @rtype: C{int}
        """
    if coordinateType is Angles.LATITUDE:
        hemisphereKey = 'latitudeHemisphere'
    elif coordinateType is Angles.LONGITUDE:
        hemisphereKey = 'longitudeHemisphere'
    elif coordinateType is Angles.VARIATION:
        hemisphereKey = 'magneticVariationDirection'
    else:
        raise ValueError(f'unknown coordinate type {coordinateType}')
    hemisphere = getattr(self.currentSentence, hemisphereKey).upper()
    if hemisphere in 'NE':
        return 1
    elif hemisphere in 'SW':
        return -1
    else:
        raise ValueError(f'bad hemisphere/direction: {hemisphere}')
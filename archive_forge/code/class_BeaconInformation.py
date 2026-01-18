from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
class BeaconInformation:
    """
    Information about positioning beacons (a generalized term for the reference
    objects that help you determine your position, such as satellites or cell
    towers).

    @ivar seenBeacons: A set of visible beacons. Note that visible beacons are not
        necessarily used in acquiring a positioning fix.
    @type seenBeacons: C{set} of L{IPositioningBeacon}
    @ivar usedBeacons: A set of the beacons that were used in obtaining a
        positioning fix. This only contains beacons that are actually used, not
        beacons for which it is unknown if they are used or not.
    @type usedBeacons: C{set} of L{IPositioningBeacon}
    """

    def __init__(self, seenBeacons=()):
        """
        Initializes a beacon information object.

        @param seenBeacons: A collection of beacons that are currently seen.
        @type seenBeacons: iterable of L{IPositioningBeacon}s
        """
        self.seenBeacons = set(seenBeacons)
        self.usedBeacons = set()

    def __repr__(self) -> str:
        """
        Returns a string representation of this beacon information object.

        The beacons are sorted by their identifier.

        @return: The string representation.
        @rtype: C{str}
        """
        sortedBeacons = partial(sorted, key=attrgetter('identifier'))
        usedBeacons = sortedBeacons(self.usedBeacons)
        unusedBeacons = sortedBeacons(self.seenBeacons - self.usedBeacons)
        template = '<BeaconInformation (used beacons ({numUsed}): {usedBeacons}, unused beacons: {unusedBeacons})>'
        formatted = template.format(numUsed=len(self.usedBeacons), usedBeacons=usedBeacons, unusedBeacons=unusedBeacons)
        return formatted
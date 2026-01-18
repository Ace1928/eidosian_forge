from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
class ClimbTests(TestCase):
    """
    Tests for L{twisted.positioning.base.Climb}.
    """

    def test_simple(self) -> None:
        """
        Speeds can be instantiated, and report their value in meters
        per second, and can be converted to floats.
        """
        climb = base.Climb(42.0)
        self.assertEqual(climb.inMetersPerSecond, 42.0)
        self.assertEqual(float(climb), 42.0)

    def test_repr(self) -> None:
        """
        Climbs report their type and value in their repr.
        """
        climb = base.Climb(42.0)
        self.assertEqual(repr(climb), '<Climb (42.0 m/s)>')

    def test_negativeClimbs(self) -> None:
        """
        Climbs can have negative values, and still report that value
        in meters per second and when converted to floats.
        """
        climb = base.Climb(-42.0)
        self.assertEqual(climb.inMetersPerSecond, -42.0)
        self.assertEqual(float(climb), -42.0)

    def test_speedInKnots(self) -> None:
        """
        A climb can be converted into its value in knots.
        """
        climb = base.Climb(1.0)
        self.assertEqual(1 / base.MPS_PER_KNOT, climb.inKnots)

    def test_asFloat(self) -> None:
        """
        A climb can be converted into a C{float}.
        """
        self.assertEqual(1.0, float(base.Climb(1.0)))
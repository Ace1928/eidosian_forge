from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
class SpeedTests(TestCase):
    """
    Tests for the L{twisted.positioning.base.Speed} class.
    """

    def test_value(self) -> None:
        """
        Speeds can be instantiated, and report their value in meters
        per second, and can be converted to floats.
        """
        speed = base.Speed(50.0)
        self.assertEqual(speed.inMetersPerSecond, 50.0)
        self.assertEqual(float(speed), 50.0)

    def test_repr(self) -> None:
        """
        Speeds report their type and value in their repr.
        """
        speed = base.Speed(50.0)
        self.assertEqual(repr(speed), '<Speed (50.0 m/s)>')

    def test_negativeSpeeds(self) -> None:
        """
        Creating a negative speed raises C{ValueError}.
        """
        self.assertRaises(ValueError, base.Speed, -1.0)

    def test_inKnots(self) -> None:
        """
        A speed can be converted into its value in knots.
        """
        speed = base.Speed(1.0)
        self.assertEqual(1 / base.MPS_PER_KNOT, speed.inKnots)

    def test_asFloat(self) -> None:
        """
        A speed can be converted into a C{float}.
        """
        self.assertEqual(1.0, float(base.Speed(1.0)))
from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
class PositioningBeaconTests(TestCase):
    """
    Tests for L{base.PositioningBeacon}.
    """

    def test_interface(self) -> None:
        """
        Tests that L{base.PositioningBeacon} implements L{IPositioningBeacon}.
        """
        implements = IPositioningBeacon.implementedBy(base.PositioningBeacon)
        self.assertTrue(implements)
        verify.verifyObject(IPositioningBeacon, base.PositioningBeacon(1))

    def test_repr(self) -> None:
        """
        Tests the repr of a positioning beacon.
        """
        self.assertEqual(repr(base.PositioningBeacon('A')), '<Beacon (A)>')
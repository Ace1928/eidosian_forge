from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
class SatelliteTests(TestCase):
    """
    Tests for L{twisted.positioning.base.Satellite}.
    """

    def test_minimal(self) -> None:
        """
        Tests a minimal satellite that only has a known PRN.

        Tests that the azimuth, elevation and signal to noise ratios
        are L{None} and verifies the repr.
        """
        s = base.Satellite(1)
        self.assertEqual(s.identifier, 1)
        self.assertIsNone(s.azimuth)
        self.assertIsNone(s.elevation)
        self.assertIsNone(s.signalToNoiseRatio)
        self.assertEqual(repr(s), '<Satellite (1), azimuth: None, elevation: None, snr: None>')

    def test_simple(self) -> None:
        """
        Tests a minimal satellite that only has a known PRN.

        Tests that the azimuth, elevation and signal to noise ratios
        are correct and verifies the repr.
        """
        s = base.Satellite(identifier=1, azimuth=270.0, elevation=30.0, signalToNoiseRatio=25.0)
        self.assertEqual(s.identifier, 1)
        self.assertEqual(s.azimuth, 270.0)
        self.assertEqual(s.elevation, 30.0)
        self.assertEqual(s.signalToNoiseRatio, 25.0)
        self.assertEqual(repr(s), '<Satellite (1), azimuth: 270.0, elevation: 30.0, snr: 25.0>')
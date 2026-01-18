from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
class BeaconInformationTests(TestCase):
    """
    Tests for L{twisted.positioning.base.BeaconInformation}.
    """

    def test_minimal(self) -> None:
        """
        For an empty beacon information object, the number of used
        beacons is zero, the number of seen beacons is zero, and the
        repr of the object reflects that.
        """
        bi = base.BeaconInformation()
        self.assertEqual(len(bi.usedBeacons), 0)
        expectedRepr = '<BeaconInformation (used beacons (0): [], unused beacons: [])>'
        self.assertEqual(repr(bi), expectedRepr)
    satelliteKwargs = {'azimuth': 1, 'elevation': 1, 'signalToNoiseRatio': 1.0}

    def test_simple(self) -> None:
        """
        Tests a beacon information with a bunch of satellites, none of
        which used in computing a fix.
        """

        def _buildSatellite(**kw: float) -> base.Satellite:
            kwargs = dict(self.satelliteKwargs)
            kwargs.update(kw)
            return base.Satellite(**kwargs)
        beacons = set()
        for prn in range(1, 10):
            beacons.add(_buildSatellite(identifier=prn))
        bi = base.BeaconInformation(beacons)
        self.assertEqual(len(bi.seenBeacons), 9)
        self.assertEqual(len(bi.usedBeacons), 0)
        self.assertEqual(repr(bi), '<BeaconInformation (used beacons (0): [], unused beacons: [<Satellite (1), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (2), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (3), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (4), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (5), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (6), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (7), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (8), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (9), azimuth: 1, elevation: 1, snr: 1.0>])>')

    def test_someSatellitesUsed(self) -> None:
        """
        Tests a beacon information with a bunch of satellites, some of
        them used in computing a fix.
        """
        bi = base.BeaconInformation()
        for prn in range(1, 10):
            satellite = base.Satellite(identifier=prn, **self.satelliteKwargs)
            bi.seenBeacons.add(satellite)
            if prn % 2:
                bi.usedBeacons.add(satellite)
        self.assertEqual(len(bi.seenBeacons), 9)
        self.assertEqual(len(bi.usedBeacons), 5)
        self.assertEqual(repr(bi), '<BeaconInformation (used beacons (5): [<Satellite (1), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (3), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (5), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (7), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (9), azimuth: 1, elevation: 1, snr: 1.0>], unused beacons: [<Satellite (2), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (4), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (6), azimuth: 1, elevation: 1, snr: 1.0>, <Satellite (8), azimuth: 1, elevation: 1, snr: 1.0>])>')
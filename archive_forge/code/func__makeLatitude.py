from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def _makeLatitude(value: float) -> base.Coordinate:
    """
    Builds and returns a latitude of given value.
    """
    return base.Coordinate(value, Angles.LATITUDE)
from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def _badValueTest(self, **kw: float) -> None:
    """
        Helper function for verifying that bad values raise C{ValueError}.

        @param kw: The keyword arguments passed to L{base.Heading.fromFloats}.
        """
    self.assertRaises(ValueError, base.Heading.fromFloats, **kw)
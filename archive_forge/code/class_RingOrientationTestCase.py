import unittest
import pytest
from shapely.geometry.polygon import LinearRing, orient, Polygon, signed_area
class RingOrientationTestCase(unittest.TestCase):

    def test_ccw(self):
        ring = LinearRing([(1, 0), (0, 1), (0, 0)])
        assert ring.is_ccw

    def test_cw(self):
        ring = LinearRing([(0, 0), (0, 1), (1, 0)])
        assert not ring.is_ccw
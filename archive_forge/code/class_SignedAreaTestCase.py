import unittest
import pytest
from shapely.geometry.polygon import LinearRing, orient, Polygon, signed_area
class SignedAreaTestCase(unittest.TestCase):

    def test_triangle(self):
        tri = LinearRing([(0, 0), (2, 5), (7, 0)])
        assert signed_area(tri) == pytest.approx(-7 * 5 / 2)

    def test_square(self):
        xmin, xmax = (-1, 1)
        ymin, ymax = (-2, 3)
        rect = LinearRing([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)])
        assert signed_area(rect) == pytest.approx(10.0)
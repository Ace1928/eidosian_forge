import unittest
from holoviews.core import AARectangle, BoundingBox
class TestBoundingBox(unittest.TestCase):

    def setUp(self):
        self.left = -0.1
        self.bottom = -0.2
        self.right = 0.3
        self.top = 0.4
        self.lbrt = (self.left, self.bottom, self.right, self.top)
        self.region = BoundingBox(points=((self.left, self.bottom), (self.right, self.top)))
        self.xc, self.yc = self.region.aarect().centroid()

    def test_way_inside(self):
        self.assertTrue(self.region.contains(0, 0))

    def test_above(self):
        self.assertFalse(self.region.contains(0, 1))

    def test_below(self):
        self.assertFalse(self.region.contains(0, -1))

    def test_left_of(self):
        self.assertFalse(self.region.contains(-1, 0))

    def test_right_of(self):
        self.assertFalse(self.region.contains(1, 0))

    def test_centroid_x(self):
        self.assertEqual(self.xc, (self.left + self.right) / 2.0)

    def test_centroid_y(self):
        self.assertEqual(self.yc, (self.bottom + self.top) / 2.0)

    def test_left_boundary(self):
        self.assertTrue(self.region.contains(self.left, self.yc))

    def test_right_boundary(self):
        self.assertTrue(self.region.contains(self.right, self.yc))

    def test_bottom_boundary(self):
        self.assertTrue(self.region.contains(self.xc, self.bottom))

    def test_top_boundary(self):
        self.assertTrue(self.region.contains(self.xc, self.top))
import unittest
from holoviews.core import AARectangle, BoundingBox
class TestAARectangle(unittest.TestCase):

    def setUp(self):
        self.left = -0.1
        self.bottom = -0.2
        self.right = 0.3
        self.top = 0.4
        self.lbrt = (self.left, self.bottom, self.right, self.top)
        self.aar1 = AARectangle((self.left, self.bottom), (self.right, self.top))
        self.aar2 = AARectangle((self.right, self.bottom), (self.left, self.top))

    def test_left(self):
        self.assertEqual(self.left, self.aar1.left())

    def test_right(self):
        self.assertEqual(self.right, self.aar1.right())

    def test_bottom(self):
        self.assertEqual(self.bottom, self.aar1.bottom())

    def test_top(self):
        self.assertEqual(self.top, self.aar1.top())

    def test_lbrt(self):
        self.assertEqual(self.lbrt, self.aar1.lbrt())

    def test_point_order(self):
        self.assertEqual(self.aar1.lbrt(), self.aar2.lbrt())
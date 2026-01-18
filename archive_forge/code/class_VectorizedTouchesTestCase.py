import unittest
import numpy as np
from shapely.geometry import box, MultiPolygon, Point
class VectorizedTouchesTestCase(unittest.TestCase):

    def test_touches(self):
        from shapely.vectorized import touches
        y, x = np.mgrid[-2:3:6j, -1:3:5j]
        geom = box(0, -1, 2, 2)
        result = touches(geom, x, y)
        expected = np.array([[False, False, False, False, False], [False, True, True, True, False], [False, True, False, True, False], [False, True, False, True, False], [False, True, True, True, False], [False, False, False, False, False]], dtype=bool)
        from numpy.testing import assert_array_equal
        assert_array_equal(result, expected)
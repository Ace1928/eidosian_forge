import gc
import weakref
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
import unittest
class TestLifeTimeIssue(TestCase):

    def test_double_free(self):
        from numba import njit
        import numpy as np

        @njit
        def is_point_in_polygons(point, polygons):
            num_polygons = polygons.shape[0]
            if num_polygons != 0:
                intentionally_unused_variable = polygons[0]
            return 0

        @njit
        def dummy():
            return np.empty(10, dtype=np.int64)
        polygons = np.array([[[0, 1]]])
        points = np.array([[-1.5, 0.5]])
        a = dummy()
        is_point_in_polygons(points[0], polygons)
        b = dummy()
        is_point_in_polygons(points[0], polygons)
        c = dummy()
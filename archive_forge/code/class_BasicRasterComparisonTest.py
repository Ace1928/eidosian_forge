import numpy as np
from holoviews import Image
from holoviews.core import BoundingBox, Dimension
from holoviews.core.element import HoloMap
from holoviews.element.comparison import ComparisonTestCase
class BasicRasterComparisonTest(RasterTestCase):
    """
    This tests the ComparisonTestCase class which is an important
    component of other tests.
    """

    def test_matrices_equal(self):
        self.assertEqual(self.mat1, self.mat1)

    def test_unequal_arrays(self):
        try:
            self.assertEqual(self.mat1, self.mat2)
            raise AssertionError('Array mismatch not raised')
        except AssertionError as e:
            if not str(e).startswith('Image not almost equal to 6 decimals\n'):
                raise self.failureException('Image data mismatch error not raised.')

    def test_bounds_mismatch(self):
        try:
            self.assertEqual(self.mat1, self.mat4)
        except AssertionError as e:
            self.assertEqual(str(e), 'BoundingBoxes are mismatched: (-0.5, -0.5, 0.5, 0.5) != (-0.3, -0.3, 0.3, 0.3).')
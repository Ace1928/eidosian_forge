import numpy as np
from holoviews.element import HSV, RGB, Curve, Image, QuadMesh, Raster
from holoviews.element.comparison import ComparisonTestCase
class TestHSV(ComparisonTestCase):

    def setUp(self):
        self.hsv_array = np.random.randint(0, 255, (3, 3, 4))

    def test_not_using_class_variables_vdims(self):
        init_vdims = HSV(self.hsv_array).vdims
        cls_vdims = HSV.vdims
        for i, c in zip(init_vdims, cls_vdims):
            assert i is not c
            assert i == c
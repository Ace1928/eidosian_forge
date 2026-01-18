import numpy as np
from holoviews.element import HSV, RGB, Curve, Image, QuadMesh, Raster
from holoviews.element.comparison import ComparisonTestCase
class TestRaster(ComparisonTestCase):

    def setUp(self):
        self.array1 = np.array([(0, 1, 2), (3, 4, 5)])

    def test_raster_init(self):
        Raster(self.array1)

    def test_raster_index(self):
        raster = Raster(self.array1)
        self.assertEqual(raster[0, 1], 3)

    def test_raster_sample(self):
        raster = Raster(self.array1)
        self.assertEqual(raster.sample(y=0), Curve(np.array([(0, 0), (1, 1), (2, 2)]), kdims=['x'], vdims=['z']))

    def test_raster_range_masked(self):
        arr = np.random.rand(10, 10) - 0.5
        arr = np.ma.masked_where(arr <= 0, arr)
        rrange = Raster(arr).range(2)
        self.assertEqual(rrange, (np.min(arr), np.max(arr)))
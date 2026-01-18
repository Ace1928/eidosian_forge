import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
class TestEllipsisRaster(ComparisonTestCase):

    def test_raster_ellipsis_slice_value(self):
        data = np.random.rand(10, 10)
        sliced = hv.Raster(data)[..., 'z']
        self.assertEqual(sliced.data, data)

    def test_raster_ellipsis_slice_value_missing(self):
        data = np.random.rand(10, 10)
        try:
            hv.Raster(data)[..., 'Non-existent']
        except Exception as e:
            if "'z' is the only selectable value dimension" not in str(e):
                raise AssertionError('Unexpected exception.')

    def test_image_ellipsis_slice_value(self):
        data = np.random.rand(10, 10)
        sliced = hv.Image(data)[..., 'z']
        self.assertEqual(sliced.data, data)

    def test_image_ellipsis_slice_value_missing(self):
        data = np.random.rand(10, 10)
        try:
            hv.Image(data)[..., 'Non-existent']
        except Exception as e:
            if str(e) != "'Non-existent' is not an available value dimension":
                raise AssertionError('Unexpected exception.')

    def test_rgb_ellipsis_slice_value(self):
        data = np.random.rand(10, 10, 3)
        sliced = hv.RGB(data)[:, :, 'R']
        self.assertEqual(sliced.data, data[:, :, 0])

    def test_rgb_ellipsis_slice_value_missing(self):
        rgb = hv.RGB(np.random.rand(10, 10, 3))
        try:
            rgb[..., 'Non-existent']
        except Exception as e:
            if str(e) != repr("'Non-existent' is not an available value dimension"):
                raise AssertionError('Incorrect exception raised.')
import numpy as np
import pytest
import cartopy.crs as ccrs
class TestOSGB:

    def setup_class(self):
        self.point_a = (-3.474083, 50.727301)
        self.point_b = (0.5, 50.5)
        self.src_crs = ccrs.PlateCarree()
        self.nan = float('nan')

    @pytest.mark.parametrize('approx', [True, False])
    def test_default(self, approx):
        proj = ccrs.OSGB(approx=approx)
        res = proj.transform_point(*self.point_a, src_crs=self.src_crs)
        np.testing.assert_array_almost_equal(res, (295971.28668, 93064.27666), decimal=5)
        res = proj.transform_point(*self.point_b, src_crs=self.src_crs)
        np.testing.assert_array_almost_equal(res, (577274.9838, 69740.49227), decimal=5)

    def test_nan(self):
        proj = ccrs.OSGB(approx=True)
        res = proj.transform_point(0.0, float('nan'), src_crs=self.src_crs)
        assert np.all(np.isnan(res))
        res = proj.transform_point(float('nan'), 0.0, src_crs=self.src_crs)
        assert np.all(np.isnan(res))
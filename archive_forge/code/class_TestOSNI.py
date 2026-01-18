import numpy as np
import pytest
import cartopy.crs as ccrs
class TestOSNI:

    def setup_class(self):
        self.point_a = (-6.826286, 54.725116)
        self.src_crs = ccrs.PlateCarree()
        self.nan = float('nan')

    @pytest.mark.parametrize('approx', [True, False])
    def test_default(self, approx):
        proj = ccrs.OSNI(approx=approx)
        res = proj.transform_point(*self.point_a, src_crs=self.src_crs)
        np.testing.assert_array_almost_equal(res, (275614.267627, 386984.20643))

    def test_nan(self):
        proj = ccrs.OSNI(approx=True)
        res = proj.transform_point(0.0, float('nan'), src_crs=self.src_crs)
        assert np.all(np.isnan(res))
        res = proj.transform_point(float('nan'), 0.0, src_crs=self.src_crs)
        assert np.all(np.isnan(res))
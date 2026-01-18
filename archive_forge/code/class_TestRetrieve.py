import numpy as np
from numpy.testing import assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.io.srtm
from .test_downloaders import download_to_temp  # noqa: F401 (used as fixture)
class TestRetrieve:

    @pytest.mark.parametrize('Source, read_SRTM, max_, min_, pt', [(cartopy.io.srtm.SRTM3Source, cartopy.io.srtm.read_SRTM3, 602, -34, 78), (cartopy.io.srtm.SRTM1Source, cartopy.io.srtm.read_SRTM1, 602, -37, 50)], ids=['srtm3', 'srtm1'])
    def test_srtm_retrieve(self, Source, read_SRTM, max_, min_, pt, download_to_temp):
        with pytest.warns(cartopy.io.DownloadWarning):
            r = Source().srtm_fname(-4, 50)
        assert r.startswith(str(download_to_temp)), 'File not downloaded to tmp dir'
        img, _, _ = read_SRTM(r)
        assert img.max() == max_
        assert img.min() == min_
        assert img[-10, 12] == pt

    @pytest.mark.parametrize('Source, shape', [(cartopy.io.srtm.SRTM3Source, (1201, 1201)), (cartopy.io.srtm.SRTM1Source, (3601, 3601))], ids=['srtm3', 'srtm1'])
    def test_srtm_out_of_range(self, Source, shape):
        img, _, _ = Source().combined(120, 2, 2, 2)
        assert_array_equal(img, np.zeros(np.array(shape) * 2))
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.io.srtm
from .test_downloaders import download_to_temp  # noqa: F401 (used as fixture)
@pytest.mark.parametrize('Source', [cartopy.io.srtm.SRTM3Source, cartopy.io.srtm.SRTM1Source], ids=['srtm3', 'srtm1'])
class TestSRTMSource__combined:

    def test_trivial(self, Source):
        source = Source()
        e_img, e_crs, e_extent = source.single_tile(-3, 50)
        r_img, r_crs, r_extent = source.combined(-3, 50, 1, 1)
        assert_array_equal(e_img, r_img)
        assert e_crs == r_crs
        assert e_extent == r_extent

    def test_2by2(self, Source):
        source = Source()
        e_img, _, e_extent = source.combined(-1, 50, 2, 1)
        assert e_extent == (-1, 1, 50, 51)
        imgs = [source.single_tile(-1, 50)[0], source.single_tile(0, 50)[0]]
        assert_array_equal(np.hstack(imgs), e_img)
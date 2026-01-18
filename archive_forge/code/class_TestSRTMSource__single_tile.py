import numpy as np
from numpy.testing import assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.io.srtm
from .test_downloaders import download_to_temp  # noqa: F401 (used as fixture)
@pytest.mark.parametrize('Source', [cartopy.io.srtm.SRTM3Source, cartopy.io.srtm.SRTM1Source], ids=['srtm3', 'srtm1'])
class TestSRTMSource__single_tile:

    def test_out_of_range(self, Source):
        source = Source()
        match = 'No srtm tile found for those coordinates\\.'
        with pytest.raises(ValueError, match=match):
            source.single_tile(-25, 50)

    def test_in_range(self, Source):
        if Source == cartopy.io.srtm.SRTM3Source:
            shape = (1201, 1201)
        elif Source == cartopy.io.srtm.SRTM1Source:
            shape = (3601, 3601)
        else:
            raise ValueError('Source is of unexpected type.')
        source = Source()
        img, crs, extent = source.single_tile(-1, 50)
        assert isinstance(img, np.ndarray)
        assert img.shape == shape
        assert img.dtype == np.dtype('>i2')
        assert crs == ccrs.PlateCarree()
        assert extent == (-1, 0, 50, 51)

    def test_zeros(self, Source):
        source = Source()
        _, _, extent = source.single_tile(0, 50)
        assert extent == (0, 1, 50, 51)
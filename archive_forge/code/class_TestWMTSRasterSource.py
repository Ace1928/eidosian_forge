from unittest import mock
from xml.etree.ElementTree import ParseError
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.io.ogc_clients as ogc
from cartopy.io.ogc_clients import _OWSLIB_AVAILABLE
@pytest.mark.filterwarnings('ignore:TileMatrixLimits')
@pytest.mark.network
@pytest.mark.skipif(not _OWSLIB_AVAILABLE, reason='OWSLib is unavailable.')
@pytest.mark.xfail(reason='NASA servers are returning bad content metadata')
class TestWMTSRasterSource:
    URI = 'https://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
    layer_name = 'VIIRS_CityLights_2012'
    projection = ccrs.PlateCarree()

    def test_string_service(self):
        source = ogc.WMTSRasterSource(self.URI, self.layer_name)
        assert isinstance(source.wmts, WebMapTileService)
        assert isinstance(source.layer, ContentMetadata)
        assert source.layer.name == self.layer_name

    def test_wmts_service_instance(self):
        service = WebMapTileService(self.URI)
        source = ogc.WMTSRasterSource(service, self.layer_name)
        assert source.wmts is service

    def test_native_projection(self):
        source = ogc.WMTSRasterSource(self.URI, self.layer_name)
        source.validate_projection(self.projection)

    def test_non_native_projection(self):
        source = ogc.WMTSRasterSource(self.URI, self.layer_name)
        source.validate_projection(ccrs.Miller())

    def test_unsupported_projection(self):
        source = ogc.WMTSRasterSource(self.URI, self.layer_name)
        with mock.patch('cartopy.io.ogc_clients._URN_TO_CRS', {}):
            match = 'Unable to find tile matrix for projection\\.'
            with pytest.raises(ValueError, match=match):
                source.validate_projection(ccrs.Miller())

    def test_fetch_img(self):
        source = ogc.WMTSRasterSource(self.URI, self.layer_name)
        extent = [-10, 10, 40, 60]
        located_image, = source.fetch_raster(self.projection, extent, RESOLUTION)
        img = np.array(located_image.image)
        assert img.shape == (512, 512, 4)
        assert img[:, :, 3].min() == 255
        assert located_image.extent == (-180.0, 107.99999999999994, -197.99999999999994, 90.0)

    def test_fetch_img_reprojected(self):
        source = ogc.WMTSRasterSource(self.URI, self.layer_name)
        extent = [-20, -1, 48, 50]
        located_image, = source.fetch_raster(ccrs.NorthPolarStereo(), extent, (30, 30))
        img = np.array(located_image.image)
        assert img.shape == (42, 42, 4)
        assert located_image.extent == extent

    def test_fetch_img_reprojected_twoparts(self):
        source = ogc.WMTSRasterSource(self.URI, self.layer_name)
        extent = [-10, 12, 48, 50]
        images = source.fetch_raster(ccrs.NorthPolarStereo(), extent, (30, 30))
        assert len(images) == 2
        im1, im2 = images
        assert np.array(im1.image).shape == (42, 42, 4)
        assert np.array(im2.image).shape == (42, 42, 4)
        assert im1.extent == extent
        assert im2.extent == extent
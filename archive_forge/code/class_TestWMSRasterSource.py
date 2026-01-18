from unittest import mock
from xml.etree.ElementTree import ParseError
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.io.ogc_clients as ogc
from cartopy.io.ogc_clients import _OWSLIB_AVAILABLE
@pytest.mark.network
@pytest.mark.skipif(not _OWSLIB_AVAILABLE, reason='OWSLib is unavailable.')
class TestWMSRasterSource:
    URI = 'http://vmap0.tiles.osgeo.org/wms/vmap0'
    layer = 'basic'
    layers = ['basic', 'ocean']
    projection = ccrs.PlateCarree()

    def test_string_service(self):
        source = ogc.WMSRasterSource(self.URI, self.layer)
        from owslib.map.wms111 import WebMapService_1_1_1
        from owslib.map.wms130 import WebMapService_1_3_0
        assert isinstance(source.service, (WebMapService_1_1_1, WebMapService_1_3_0))
        assert isinstance(source.layers, list)
        assert source.layers == [self.layer]

    def test_wms_service_instance(self):
        service = WebMapService(self.URI)
        source = ogc.WMSRasterSource(service, self.layer)
        assert source.service is service

    def test_multiple_layers(self):
        source = ogc.WMSRasterSource(self.URI, self.layers)
        assert source.layers == self.layers

    def test_no_layers(self):
        match = 'One or more layers must be defined\\.'
        with pytest.raises(ValueError, match=match):
            ogc.WMSRasterSource(self.URI, [])

    def test_extra_kwargs_empty(self):
        source = ogc.WMSRasterSource(self.URI, self.layer, getmap_extra_kwargs={})
        assert source.getmap_extra_kwargs == {}

    def test_extra_kwargs_None(self):
        source = ogc.WMSRasterSource(self.URI, self.layer, getmap_extra_kwargs=None)
        assert source.getmap_extra_kwargs == {'transparent': True}

    def test_extra_kwargs_non_empty(self):
        kwargs = {'another': 'kwarg'}
        source = ogc.WMSRasterSource(self.URI, self.layer, getmap_extra_kwargs=kwargs)
        assert source.getmap_extra_kwargs == kwargs

    def test_supported_projection(self):
        source = ogc.WMSRasterSource(self.URI, self.layer)
        source.validate_projection(self.projection)

    def test_unsupported_projection(self):
        source = ogc.WMSRasterSource(self.URI, self.layer)
        with mock.patch.dict('cartopy.io.ogc_clients._CRS_TO_OGC_SRS', {ccrs.OSNI(approx=True): 'EPSG:29901'}, clear=True):
            msg = 'not available'
            with pytest.raises(ValueError, match=msg):
                source.validate_projection(ccrs.Miller())

    def test_fetch_img(self):
        source = ogc.WMSRasterSource(self.URI, self.layer)
        extent = [-10, 10, 40, 60]
        located_image, = source.fetch_raster(self.projection, extent, RESOLUTION)
        img = np.array(located_image.image)
        assert img.shape == RESOLUTION + (4,)
        assert img[:, :, 3].min() == 255
        assert extent == located_image.extent

    def test_fetch_img_different_projection(self):
        source = ogc.WMSRasterSource(self.URI, self.layer)
        extent = [-570000, 5100000, 870000, 3500000]
        located_image, = source.fetch_raster(ccrs.Orthographic(), extent, RESOLUTION)
        img = np.array(located_image.image)
        assert img.shape == RESOLUTION + (4,)

    def test_multi_image_result(self):
        source = ogc.WMSRasterSource(self.URI, self.layer)
        crs = ccrs.PlateCarree(central_longitude=180)
        extent = [-15, 25, 45, 85]
        located_images = source.fetch_raster(crs, extent, RESOLUTION)
        assert len(located_images) == 2

    def test_float_resolution(self):
        source = ogc.WMSRasterSource(self.URI, self.layer)
        extent = [-570000, 5100000, 870000, 3500000]
        located_image, = source.fetch_raster(self.projection, extent, [19.5, 39.1])
        img = np.array(located_image.image)
        assert img.shape == (40, 20, 4)
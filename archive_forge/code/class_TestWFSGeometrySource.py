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
class TestWFSGeometrySource:
    URI = 'https://nsidc.org/cgi-bin/atlas_south?service=WFS'
    typename = 'land_excluding_antarctica'
    native_projection = ccrs.Stereographic(central_latitude=-90, true_scale_latitude=-71)

    def test_string_service(self):
        service = WebFeatureService(self.URI)
        source = ogc.WFSGeometrySource(self.URI, self.typename)
        assert isinstance(source.service, type(service))
        assert source.features == [self.typename]

    def test_wfs_service_instance(self):
        service = WebFeatureService(self.URI)
        source = ogc.WFSGeometrySource(service, self.typename)
        assert source.service is service
        assert source.features == [self.typename]

    def test_default_projection(self):
        source = ogc.WFSGeometrySource(self.URI, self.typename)
        assert source.default_projection() == self.native_projection

    def test_unsupported_projection(self):
        source = ogc.WFSGeometrySource(self.URI, self.typename)
        msg = 'Geometries are only available in projection'
        with pytest.raises(ValueError, match=msg):
            source.fetch_geometries(ccrs.PlateCarree(), [-180, 180, -90, 90])

    @pytest.mark.xfail(raises=ParseError, reason='Bad XML returned from the URL')
    def test_fetch_geometries(self):
        source = ogc.WFSGeometrySource(self.URI, self.typename)
        extent = (-99012, 1523166, -6740315, -4589165)
        geoms = source.fetch_geometries(self.native_projection, extent)
        assert len(geoms) == 23
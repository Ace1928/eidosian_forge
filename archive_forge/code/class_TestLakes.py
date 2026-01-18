from pathlib import Path
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import shapely.geometry as sgeom
import cartopy.io.shapereader as shp
class TestLakes:

    @pytest.fixture(autouse=True, params=[0, 1])
    def setup_class(self, request):
        LAKES_PATH = Path(__file__).parent / 'lakes_shapefile' / 'ne_110m_lakes.shp'
        if request.param == 0:
            self.reader = shp.BasicReader(LAKES_PATH)
        elif not shp._HAS_FIONA:
            pytest.skip('Fiona library not available')
        else:
            self.reader = shp.FionaReader(LAKES_PATH)
        names = [record.attributes['name'] for record in self.reader.records()]
        self.lake_name = 'Lago de\rNicaragua'
        self.lake_index = names.index(self.lake_name)
        self.test_lake_geometry = list(self.reader.geometries())[self.lake_index]
        self.test_lake_record = list(self.reader.records())[self.lake_index]

    def test_geometry(self):
        lake_geometry = self.test_lake_geometry
        assert lake_geometry.geom_type == 'Polygon'
        polygon = sgeom.polygon.orient(lake_geometry, -1)
        expected = np.array([(-84.85548682324658, 11.147898667846633), (-85.29013729525353, 11.176165676310276), (-85.79132117383625, 11.509737046754324), (-85.8851655748783, 11.900100816287136), (-85.5653401354239, 11.940330918826362), (-85.03684526237491, 11.5216484643976), (-84.85548682324658, 11.147898667846633), (-84.85548682324658, 11.147898667846633)])
        assert_array_almost_equal(expected, polygon.exterior.coords)
        assert len(polygon.interiors) == 0

    def test_record(self):
        lake_record = self.test_lake_record
        assert lake_record.attributes.get('name') == self.lake_name
        expected = sorted(['admin', 'featurecla', 'min_label', 'min_zoom', 'name', 'name_alt', 'scalerank'])
        actual = sorted(lake_record.attributes.keys())
        assert actual == expected
        assert lake_record.geometry == self.test_lake_geometry

    def test_bounds(self):
        if isinstance(self.reader, shp.BasicReader):
            record = next(self.reader.records())
            assert not record._geometry, 'The geometry was loaded before it was needed.'
            assert len(record._bounds) == 4
            assert record._bounds == record.bounds
            assert not record._geometry, 'The geometry was loaded in order to create the bounds.'
        else:
            pytest.skip("Fiona reader doesn't support lazy loading")
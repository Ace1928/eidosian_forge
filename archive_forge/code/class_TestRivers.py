from pathlib import Path
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import shapely.geometry as sgeom
import cartopy.io.shapereader as shp
@pytest.mark.filterwarnings('ignore:Downloading')
@pytest.mark.natural_earth
class TestRivers:

    def setup_class(self):
        RIVERS_PATH = shp.natural_earth(resolution='110m', category='physical', name='rivers_lake_centerlines')
        self.reader = shp.Reader(RIVERS_PATH)
        names = [record.attributes['name'] for record in self.reader.records()]
        self.river_name = 'Peace'
        self.river_index = names.index(self.river_name)
        self.test_river_geometry = list(self.reader.geometries())[self.river_index]
        self.test_river_record = list(self.reader.records())[self.river_index]

    def test_geometry(self):
        geometry = self.test_river_geometry
        assert geometry.geom_type == 'LineString'
        linestring = geometry
        coords = linestring.coords
        assert round(abs(coords[0][0] - -124.83563045947423), 7) == 0
        assert round(abs(coords[0][1] - 56.75692352968272), 7) == 0
        assert round(abs(coords[1][0] - -124.20045039940291), 7) == 0
        assert round(abs(coords[1][1] - 56.243492336646824), 7) == 0

    def test_record(self):
        records = list(self.reader.records())
        assert len(records) == len(self.reader)
        river_record = records[self.river_index]
        expected_attributes = {'featurecla': 'River', 'min_label': 3.1, 'min_zoom': 2.1, 'name': self.river_name, 'name_en': self.river_name, 'scalerank': 2}
        for key, value in river_record.attributes.items():
            if key in expected_attributes:
                assert value == expected_attributes[key]
        assert river_record.geometry == self.test_river_geometry
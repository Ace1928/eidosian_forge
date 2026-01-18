import pytest
import cartopy
import cartopy.io.shapereader as shp
@pytest.mark.filterwarnings('ignore:Downloading')
@pytest.mark.natural_earth
class TestCoastline:

    def test_robust(self):
        COASTLINE_PATH = shp.natural_earth()
        projection = cartopy.crs.TransverseMercator(central_longitude=-90, approx=False)
        reader = shp.Reader(COASTLINE_PATH)
        all_geometries = list(reader.geometries())
        geometries = []
        geometries += all_geometries
        for geometry in geometries[93:]:
            projection.project_geometry(geometry)
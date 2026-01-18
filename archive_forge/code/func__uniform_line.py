from warnings import warn
import numpy
from shapely.geometry import MultiPoint
from geopandas.array import from_shapely, points_from_xy
from geopandas.geoseries import GeoSeries
def _uniform_line(geom, size, generator):
    """
    Sample points from an input shapely linestring
    """
    fracs = generator.uniform(size=size)
    return from_shapely(geom.interpolate(fracs, normalized=True)).unary_union()
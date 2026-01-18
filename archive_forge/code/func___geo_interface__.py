import shapely
from shapely.errors import EmptyPartError
from shapely.geometry import point
from shapely.geometry.base import BaseMultipartGeometry
@property
def __geo_interface__(self):
    return {'type': 'MultiPoint', 'coordinates': tuple((g.coords[0] for g in self.geoms))}
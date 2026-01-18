import numpy as np
from shapely import Geometry, GeometryType, lib
from shapely._geometry_helpers import collections_1d, simple_geometries_1d
from shapely.decorators import multithreading_enabled
from shapely.io import from_wkt
def _xyz_to_coords(x, y, z):
    if y is None:
        return x
    if z is None:
        coords = np.broadcast_arrays(x, y)
    else:
        coords = np.broadcast_arrays(x, y, z)
    return np.stack(coords, axis=-1)
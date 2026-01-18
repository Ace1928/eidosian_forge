import numpy as np
from shapely import creation
from shapely._geometry import (
from shapely.coordinates import get_coordinates
from shapely.predicates import is_empty, is_missing
def _point_from_flatcoords(coords):
    result = creation.points(coords)
    empties = np.isnan(coords).all(axis=1)
    if empties.any():
        result[empties] = creation.empty(1, geom_type=GeometryType.POINT).item()
    return result
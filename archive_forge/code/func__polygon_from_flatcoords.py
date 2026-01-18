import numpy as np
from shapely import creation
from shapely._geometry import (
from shapely.coordinates import get_coordinates
from shapely.predicates import is_empty, is_missing
def _polygon_from_flatcoords(coords, offsets1, offsets2):
    ring_lengths = np.diff(offsets1)
    ring_indices = np.repeat(np.arange(len(ring_lengths)), ring_lengths)
    rings = creation.linearrings(coords, indices=ring_indices)
    polygon_rings_n = np.diff(offsets2)
    polygon_indices = np.repeat(np.arange(len(polygon_rings_n)), polygon_rings_n)
    result = np.empty(len(offsets2) - 1, dtype=object)
    result = creation.polygons(rings, indices=polygon_indices, out=result)
    result[polygon_rings_n == 0] = creation.empty(1, geom_type=GeometryType.POLYGON).item()
    return result
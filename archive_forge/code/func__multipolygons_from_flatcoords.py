import numpy as np
from shapely import creation
from shapely._geometry import (
from shapely.coordinates import get_coordinates
from shapely.predicates import is_empty, is_missing
def _multipolygons_from_flatcoords(coords, offsets1, offsets2, offsets3):
    polygons = _polygon_from_flatcoords(coords, offsets1, offsets2)
    multipolygon_parts = np.diff(offsets3)
    multipolygon_indices = np.repeat(np.arange(len(multipolygon_parts)), multipolygon_parts)
    result = np.empty(len(offsets3) - 1, dtype=object)
    result = creation.multipolygons(polygons, indices=multipolygon_indices, out=result)
    result[multipolygon_parts == 0] = creation.empty(1, geom_type=GeometryType.MULTIPOLYGON).item()
    return result
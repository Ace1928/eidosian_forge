import numpy as np
from shapely import creation
from shapely._geometry import (
from shapely.coordinates import get_coordinates
from shapely.predicates import is_empty, is_missing
def _multilinestrings_from_flatcoords(coords, offsets1, offsets2):
    linestrings = _linestring_from_flatcoords(coords, offsets1)
    multilinestring_parts = np.diff(offsets2)
    multilinestring_indices = np.repeat(np.arange(len(multilinestring_parts)), multilinestring_parts)
    result = np.empty(len(offsets2) - 1, dtype=object)
    result = creation.multilinestrings(linestrings, indices=multilinestring_indices, out=result)
    result[multilinestring_parts == 0] = creation.empty(1, geom_type=GeometryType.MULTILINESTRING).item()
    return result
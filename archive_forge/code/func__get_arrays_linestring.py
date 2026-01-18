import numpy as np
from shapely import creation
from shapely._geometry import (
from shapely.coordinates import get_coordinates
from shapely.predicates import is_empty, is_missing
def _get_arrays_linestring(arr, include_z):
    coords, indices = get_coordinates(arr, return_index=True, include_z=include_z)
    offsets = _indices_to_offsets(indices, len(arr))
    return (coords, (offsets,))
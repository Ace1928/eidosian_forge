import numpy as np
from shapely import creation
from shapely._geometry import (
from shapely.coordinates import get_coordinates
from shapely.predicates import is_empty, is_missing
def _get_arrays_polygon(arr, include_z):
    arr_flat, ring_indices = get_rings(arr, return_index=True)
    offsets2 = _indices_to_offsets(ring_indices, len(arr))
    coords, indices = get_coordinates(arr_flat, return_index=True, include_z=include_z)
    offsets1 = np.insert(np.bincount(indices).cumsum(), 0, 0)
    return (coords, (offsets1, offsets2))
import numpy as np
from shapely import creation
from shapely._geometry import (
from shapely.coordinates import get_coordinates
from shapely.predicates import is_empty, is_missing
def _get_arrays_multipoint(arr, include_z):
    _, part_indices = get_parts(arr, return_index=True)
    offsets = _indices_to_offsets(part_indices, len(arr))
    coords = get_coordinates(arr, include_z=include_z)
    return (coords, (offsets,))
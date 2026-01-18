import numpy as np
def _hilbert_distance(geoms, total_bounds=None, level=16):
    """
    Calculate the distance along a Hilbert curve.

    The distances are calculated for the midpoints of the geometries in the
    GeoDataFrame.

    Parameters
    ----------
    geoms : GeometryArray
    total_bounds : 4-element array
        Total bounds of geometries - array
    level : int (1 - 16), default 16
        Determines the precision of the curve (points on the curve will
        have coordinates in the range [0, 2^level - 1]).

    Returns
    -------
    np.ndarray
        Array containing distances along the Hilbert curve

    """
    if geoms.is_empty.any() | geoms.isna().any():
        raise ValueError('Hilbert distance cannot be computed on a GeoSeries with empty or missing geometries.')
    bounds = geoms.bounds
    x, y = _continuous_to_discrete_coords(bounds, level, total_bounds)
    distances = _encode(level, x, y)
    return distances
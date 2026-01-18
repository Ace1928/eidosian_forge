from itertools import product
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from ..measure.pnpoly import grid_points_in_poly
from ._convex_hull import possible_hull
from ..measure._label import label
from ..util import unique_rows
from .._shared.utils import warn
def _check_coords_in_hull(gridcoords, hull_equations, tolerance):
    """Checks all the coordinates for inclusiveness in the convex hull.

    Parameters
    ----------
    gridcoords : (M, N) ndarray
        Coordinates of ``N`` points in ``M`` dimensions.
    hull_equations : (M, N) ndarray
        Hyperplane equations of the facets of the convex hull.
    tolerance : float
        Tolerance when determining whether a point is inside the hull. Due
        to numerical floating point errors, a tolerance of 0 can result in
        some points erroneously being classified as being outside the hull.

    Returns
    -------
    coords_in_hull : ndarray of bool
        Binary 1D ndarray representing points in n-dimensional space
        with value ``True`` set for points inside the convex hull.

    Notes
    -----
    Checking the inclusiveness of coordinates in a convex hull requires
    intermediate calculations of dot products which are memory-intensive.
    Thus, the convex hull equations are checked individually with all
    coordinates to keep within the memory limit.

    References
    ----------
    .. [1] https://github.com/scikit-image/scikit-image/issues/5019

    """
    ndim, n_coords = gridcoords.shape
    n_hull_equations = hull_equations.shape[0]
    coords_in_hull = np.ones(n_coords, dtype=bool)
    dot_array = np.empty(n_coords, dtype=np.float64)
    test_ineq_temp = np.empty(n_coords, dtype=np.float64)
    coords_single_ineq = np.empty(n_coords, dtype=bool)
    for idx in range(n_hull_equations):
        np.dot(hull_equations[idx, :ndim], gridcoords, out=dot_array)
        np.add(dot_array, hull_equations[idx, ndim:], out=test_ineq_temp)
        np.less(test_ineq_temp, tolerance, out=coords_single_ineq)
        coords_in_hull *= coords_single_ineq
    return coords_in_hull
from .line import R13LineWithMatrix
from ..verify.shapes import compute_hyperbolic_shapes # type: ignore
from ..snap.fundamental_polyhedron import FundamentalPolyhedronEngine # type: ignore
from ..snap.kernel_structures import TransferKernelStructuresEngine # type: ignore
from ..snap.t3mlite import simplex, Mcomplex, Tetrahedron, Vertex # type: ignore
from ..SnapPy import word_as_list # type: ignore
from ..hyperboloid import (o13_inverse,  # type: ignore
from ..upper_halfspace import sl2c_inverse, psl2c_to_o13 # type: ignore
from ..upper_halfspace.ideal_point import ideal_point_to_r13 # type: ignore
from ..matrix import vector, matrix, mat_solve # type: ignore
from ..math_basics import prod, xgcd # type: ignore
from collections import deque
from typing import Tuple, Sequence, Optional, Any
def _filling_matrix(cusp_info: dict) -> FillingMatrix:
    """
    Given one of the dictionaries returned by Manifold.cusp_info(),
    returns the "filling matrix" filling_matrix.

    filling_matrix is a matrix of integers (as list of lists) such that
    filling_matrix[0] contains the filling coefficients
    (e.g., [3,4] for m004(3,4)) and the determinant is 1 if the cusp is
    filled. That is, filling_matrix[1] determines a curve intersecting
    the filling curve once (as sum of a multiple of meridian and
    longitude) and that is thus parallel to the core curve.

    For an unfilled cusp, filling_matrix is ((0,0), (0,0))

    Raises an exception if the filling coefficients are non-integral or
    not coprime.
    """
    float_m, float_l = cusp_info['filling']
    m = int(float_m)
    l = int(float_l)
    if float_m != m or float_l != l:
        raise ValueError('Filling coefficients (%r,%r) are not integral.' % (float_m, float_l))
    if (m, l) == (0, 0):
        return ((0, 0), (0, 0))
    n, a, b = xgcd(m, l)
    if n != 1:
        raise ValueError('Filling coefficients (%d,%d) are not co-prime.' % (m, l))
    return ((m, l), (-b, a))
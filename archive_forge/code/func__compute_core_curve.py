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
def _compute_core_curve(mcomplex: Mcomplex, peripheral_words: Sequence[Sequence[int]], core_curve_coefficients: Filling) -> R13LineWithMatrix:
    """
    Compute core curve given words for meridian and longitude and
    the integers determining a curve (as sum of a multiple of meridian
    and longitude) that is parallel to the core curve.
    """
    result = mcomplex.GeneratorMatrices[0]
    for word, f in zip(peripheral_words, core_curve_coefficients):
        if f != 0:
            m = word_list_to_psl2c_matrix(mcomplex, word)
            if f < 0:
                m = sl2c_inverse(m)
            for i in range(abs(f)):
                result = result * m
    return R13LineWithMatrix.from_psl2c_matrix(result)
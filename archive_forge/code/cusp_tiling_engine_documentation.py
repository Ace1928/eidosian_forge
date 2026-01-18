from ...sage_helper import _within_sage
from ...math_basics import correct_max
from ...snap.kernel_structures import *
from ...snap.fundamental_polyhedron import *
from ...snap.mcomplex_base import *
from ...snap.t3mlite import simplex
from ...snap import t3mlite as t3m
from ...exceptions import InsufficientPrecisionError
from ..cuspCrossSection import ComplexCuspCrossSection
from ..upper_halfspace.ideal_point import *
from ..interval_tree import *
from .cusp_translate_engine import *
import heapq

    Test::

        sage: from snappy import *
        sage: M = Manifold("s776")
        sage: C = CuspTilingEngine.from_manifold_and_shapes(
        ...     M, M.verify_hyperbolicity()[1])
        sage: C.compute_maximal_cusp_area_matrix_row(0) # doctest: +NUMERIC6
        [28.000000000?, 7.000000000?, 7.0000000000?]
        sage: C.compute_maximal_cusp_area_matrix_row(1) # doctest: +NUMERIC6
        [7.0000000000?, 28.00000000?, 7.0000000000?]
    
from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core import S, Dummy, Lambda
from sympy.core.symbol import Str
from sympy.core.symbol import symbols
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.matrices.matrices import MatrixBase
from sympy.solvers import solve
from sympy.vector.scalar import BaseScalar
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)
from sympy.matrices.dense import eye
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
import sympy.vector
from sympy.vector.orienters import (Orienter, AxisOrienter, BodyOrienter,
from sympy.vector.vector import BaseVector
@staticmethod
def _calculate_lame_coeff(equations):
    """
        It calculates Lame coefficients
        for given transformations equations.

        Parameters
        ==========

        equations : Lambda
            Lambda of transformation equations.

        """
    return lambda x1, x2, x3: (sqrt(diff(equations(x1, x2, x3)[0], x1) ** 2 + diff(equations(x1, x2, x3)[1], x1) ** 2 + diff(equations(x1, x2, x3)[2], x1) ** 2), sqrt(diff(equations(x1, x2, x3)[0], x2) ** 2 + diff(equations(x1, x2, x3)[1], x2) ** 2 + diff(equations(x1, x2, x3)[2], x2) ** 2), sqrt(diff(equations(x1, x2, x3)[0], x3) ** 2 + diff(equations(x1, x2, x3)[1], x3) ** 2 + diff(equations(x1, x2, x3)[2], x3) ** 2))
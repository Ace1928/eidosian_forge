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
def _calculate_inv_trans_equations(self):
    """
        Helper method for set_coordinate_type. It calculates inverse
        transformation equations for given transformations equations.

        """
    x1, x2, x3 = symbols('x1, x2, x3', cls=Dummy, reals=True)
    x, y, z = symbols('x, y, z', cls=Dummy)
    equations = self._transformation(x1, x2, x3)
    solved = solve([equations[0] - x, equations[1] - y, equations[2] - z], (x1, x2, x3), dict=True)[0]
    solved = (solved[x1], solved[x2], solved[x3])
    self._transformation_from_parent_lambda = lambda x1, x2, x3: tuple((i.subs(list(zip((x, y, z), (x1, x2, x3)))) for i in solved))
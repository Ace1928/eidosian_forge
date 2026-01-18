import random
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from .common import ShapeError
from .decompositions import _cholesky, _LDLdecomposition
from .matrices import MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix
from .solvers import _lower_triangular_solve, _upper_triangular_solve
@property
def _mat(self):
    sympy_deprecation_warning('\n            The private _mat attribute of Matrix is deprecated. Use the\n            .flat() method instead.\n            ', deprecated_since_version='1.9', active_deprecations_target='deprecated-private-matrix-attributes')
    return self.flat()
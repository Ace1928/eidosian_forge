from sympy.matrices.common import _MinimalMatrix, _CastableMatrix
from sympy.matrices.matrices import MatrixSubspaces
from sympy.matrices import Matrix
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.solvers import solve
class SubspaceOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixSubspaces):
    pass
from sympy.logic.boolalg import conjuncts
from sympy.assumptions import Q, ask
from sympy.assumptions.handlers import test_closed_group
from sympy.matrices import MatrixBase
from sympy.matrices.expressions import (BlockMatrix, BlockDiagMatrix, Determinant,
from sympy.matrices.expressions.blockmatrix import reblock_2x2
from sympy.matrices.expressions.factorizations import Factorization
from sympy.matrices.expressions.fourier import DFT
from sympy.core.logic import fuzzy_and
from sympy.utilities.iterables import sift
from sympy.core import Basic
from ..predicates.matrices import (SquarePredicate, SymmetricPredicate,
def MatMul_elements(matrix_predicate, scalar_predicate, expr, assumptions):
    d = sift(expr.args, lambda x: isinstance(x, MatrixExpr))
    factors, matrices = (d[False], d[True])
    return fuzzy_and([test_closed_group(Basic(*factors), assumptions, scalar_predicate), test_closed_group(Basic(*matrices), assumptions, matrix_predicate)])
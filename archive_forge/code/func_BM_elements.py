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
def BM_elements(predicate, expr, assumptions):
    """ Block Matrix elements. """
    return all((ask(predicate(b), assumptions) for b in expr.blocks))
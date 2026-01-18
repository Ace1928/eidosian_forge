from sympy.matrices.expressions import MatrixExpr
from sympy.assumptions.ask import Q
class UofLU(Factorization):

    @property
    def predicates(self):
        return (Q.upper_triangular,)
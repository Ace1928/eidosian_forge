import itertools
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand as _expand
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.common import ShapeError
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.stats.rv import RandomSymbol, is_random
from sympy.core.sympify import _sympify
from sympy.stats.symbolic_probability import Variance, Covariance, Expectation
class CrossCovarianceMatrix(Covariance, MatrixExpr):
    """
    Covariance of a random matrix probability expression.

    Examples
    ========

    >>> from sympy.stats import CrossCovarianceMatrix
    >>> from sympy.stats.rv import RandomMatrixSymbol
    >>> from sympy import symbols, MatrixSymbol
    >>> k = symbols("k")
    >>> A, B = MatrixSymbol("A", k, k), MatrixSymbol("B", k, k)
    >>> C, D = MatrixSymbol("C", k, k), MatrixSymbol("D", k, k)
    >>> X, Y = RandomMatrixSymbol("X", k, 1), RandomMatrixSymbol("Y", k, 1)
    >>> Z, W = RandomMatrixSymbol("Z", k, 1), RandomMatrixSymbol("W", k, 1)
    >>> CrossCovarianceMatrix(X, Y)
    CrossCovarianceMatrix(X, Y)
    >>> CrossCovarianceMatrix(X, Y).shape
    (k, k)

    To expand the covariance in its expression, use ``expand()``:

    >>> CrossCovarianceMatrix(X + Y, Z).expand()
    CrossCovarianceMatrix(X, Z) + CrossCovarianceMatrix(Y, Z)
    >>> CrossCovarianceMatrix(A*X, Y).expand()
    A*CrossCovarianceMatrix(X, Y)
    >>> CrossCovarianceMatrix(A*X, B.T*Y).expand()
    A*CrossCovarianceMatrix(X, Y)*B
    >>> CrossCovarianceMatrix(A*X + B*Y, C.T*Z + D.T*W).expand()
    A*CrossCovarianceMatrix(X, W)*D + A*CrossCovarianceMatrix(X, Z)*C + B*CrossCovarianceMatrix(Y, W)*D + B*CrossCovarianceMatrix(Y, Z)*C

    """

    def __new__(cls, arg1, arg2, condition=None):
        arg1 = _sympify(arg1)
        arg2 = _sympify(arg2)
        if 1 not in arg1.shape or 1 not in arg2.shape or arg1.shape[1] != arg2.shape[1]:
            raise ShapeError('Expression is not a vector')
        shape = (arg1.shape[0], arg2.shape[0]) if arg1.shape[1] == 1 and arg2.shape[1] == 1 else (1, 1)
        if condition:
            obj = Expr.__new__(cls, arg1, arg2, condition)
        else:
            obj = Expr.__new__(cls, arg1, arg2)
        obj._shape = shape
        obj._condition = condition
        return obj

    @property
    def shape(self):
        return self._shape

    def expand(self, **hints):
        arg1 = self.args[0]
        arg2 = self.args[1]
        condition = self._condition
        if arg1 == arg2:
            return VarianceMatrix(arg1, condition).expand()
        if not is_random(arg1) or not is_random(arg2):
            return ZeroMatrix(*self.shape)
        if isinstance(arg1, RandomSymbol) and isinstance(arg2, RandomSymbol):
            return CrossCovarianceMatrix(arg1, arg2, condition)
        coeff_rv_list1 = self._expand_single_argument(arg1.expand())
        coeff_rv_list2 = self._expand_single_argument(arg2.expand())
        addends = [a * CrossCovarianceMatrix(r1, r2, condition=condition) * b.transpose() for a, r1 in coeff_rv_list1 for b, r2 in coeff_rv_list2]
        return Add.fromiter(addends)

    @classmethod
    def _expand_single_argument(cls, expr):
        if isinstance(expr, RandomSymbol):
            return [(S.One, expr)]
        elif isinstance(expr, Add):
            outval = []
            for a in expr.args:
                if isinstance(a, (Mul, MatMul)):
                    outval.append(cls._get_mul_nonrv_rv_tuple(a))
                elif is_random(a):
                    outval.append((S.One, a))
            return outval
        elif isinstance(expr, (Mul, MatMul)):
            return [cls._get_mul_nonrv_rv_tuple(expr)]
        elif is_random(expr):
            return [(S.One, expr)]

    @classmethod
    def _get_mul_nonrv_rv_tuple(cls, m):
        rv = []
        nonrv = []
        for a in m.args:
            if is_random(a):
                rv.append(a)
            else:
                nonrv.append(a)
        return (Mul.fromiter(nonrv), Mul.fromiter(rv))
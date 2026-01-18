from __future__ import annotations
from sympy.core.expr import Expr
from sympy.core.function import Derivative
from sympy.core.numbers import Integer
from sympy.matrices.common import MatrixCommon
from .ndim_array import NDimArray
from .arrayop import derive_by_array
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.matrices.expressions.matexpr import _matrix_derivative
@staticmethod
def _call_derive_array_by_scalar(expr: NDimArray, v: Expr) -> Expr:
    return expr.applyfunc(lambda x: x.diff(v))
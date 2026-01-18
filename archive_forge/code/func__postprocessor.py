from __future__ import annotations
from functools import wraps
from sympy.core import S, Integer, Basic, Mul, Add
from sympy.core.assumptions import check_assumptions
from sympy.core.decorators import call_highest_priority
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.logic import FuzzyBool
from sympy.core.symbol import Str, Dummy, symbols, Symbol
from sympy.core.sympify import SympifyError, _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.matrices import MatrixKind, MatrixBase
from sympy.multipledispatch import dispatch
from sympy.utilities.misc import filldedent
from .matmul import MatMul
from .matadd import MatAdd
from .matpow import MatPow
from .transpose import Transpose
from .inverse import Inverse
from .special import ZeroMatrix, Identity
from .determinant import Determinant
def _postprocessor(expr):
    mat_class = {Mul: MatMul, Add: MatAdd}[cls]
    nonmatrices = []
    matrices = []
    for term in expr.args:
        if isinstance(term, MatrixExpr):
            matrices.append(term)
        else:
            nonmatrices.append(term)
    if not matrices:
        return cls._from_args(nonmatrices)
    if nonmatrices:
        if cls == Mul:
            for i in range(len(matrices)):
                if not matrices[i].is_MatrixExpr:
                    matrices[i] = matrices[i].__mul__(cls._from_args(nonmatrices))
                    nonmatrices = []
                    break
        else:
            return cls._from_args(nonmatrices + [mat_class(*matrices).doit(deep=False)])
    if mat_class == MatAdd:
        return mat_class(*matrices).doit(deep=False)
    return mat_class(cls._from_args(nonmatrices), *matrices).doit(deep=False)
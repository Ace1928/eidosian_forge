from __future__ import annotations
from typing import Any
from functools import reduce
from itertools import permutations
from sympy.combinatorics import Permutation
from sympy.core import (
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol, Dummy
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions import factorial
from sympy.matrices import ImmutableDenseMatrix as Matrix
from sympy.solvers import solve
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.simplify.simplify import simplify
class LieDerivative(Expr):
    """Lie derivative with respect to a vector field.

    Explanation
    ===========

    The transport operator that defines the Lie derivative is the pushforward of
    the field to be derived along the integral curve of the field with respect
    to which one derives.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r, R2_p
    >>> from sympy.diffgeom import (LieDerivative, TensorProduct)

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> e_rho, e_theta = R2_p.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> LieDerivative(e_x, fy)
    0
    >>> LieDerivative(e_x, fx)
    1
    >>> LieDerivative(e_x, e_x)
    0

    The Lie derivative of a tensor field by another tensor field is equal to
    their commutator:

    >>> LieDerivative(e_x, e_rho)
    Commutator(e_x, e_rho)
    >>> LieDerivative(e_x + e_y, fx)
    1

    >>> tp = TensorProduct(dx, dy)
    >>> LieDerivative(e_x, tp)
    LieDerivative(e_x, TensorProduct(dx, dy))
    >>> LieDerivative(e_x, tp)
    LieDerivative(e_x, TensorProduct(dx, dy))

    """

    def __new__(cls, v_field, expr):
        expr_form_ord = covariant_order(expr)
        if contravariant_order(v_field) != 1 or covariant_order(v_field):
            raise ValueError('Lie derivatives are defined only with respect to vector fields. The supplied argument was not a vector field.')
        if expr_form_ord > 0:
            obj = super().__new__(cls, v_field, expr)
            obj._v_field = v_field
            obj._expr = expr
            return obj
        if expr.atoms(BaseVectorField):
            return Commutator(v_field, expr)
        else:
            return v_field.rcall(expr)

    @property
    def v_field(self):
        return self.args[0]

    @property
    def expr(self):
        return self.args[1]

    def __call__(self, *args):
        v = self.v_field
        expr = self.expr
        lead_term = v(expr(*args))
        rest = Add(*[Mul(*args[:i] + (Commutator(v, args[i]),) + args[i + 1:]) for i in range(len(args))])
        return lead_term - rest
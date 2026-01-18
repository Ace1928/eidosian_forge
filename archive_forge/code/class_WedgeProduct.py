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
class WedgeProduct(TensorProduct):
    """Wedge product of forms.

    Explanation
    ===========

    In the context of integration only completely antisymmetric forms make
    sense. The wedge product permits the creation of such forms.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import WedgeProduct

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> WedgeProduct(dx, dy)(e_x, e_y)
    1
    >>> WedgeProduct(dx, dy)(e_y, e_x)
    -1
    >>> WedgeProduct(dx, fx*dy)(fx*e_x, e_y)
    x**2
    >>> WedgeProduct(e_x, e_y)(fy, None)
    -e_x

    You can nest wedge products.

    >>> wp1 = WedgeProduct(dx, dy)
    >>> WedgeProduct(wp1, dx)(e_x, e_y, e_x)
    0

    """

    def __call__(self, *fields):
        """Apply on a list of vector_fields.
        The expression is rewritten internally in terms of tensor products and evaluated."""
        orders = (covariant_order(e) + contravariant_order(e) for e in self.args)
        mul = 1 / Mul(*(factorial(o) for o in orders))
        perms = permutations(fields)
        perms_par = (Permutation(p).signature() for p in permutations(range(len(fields))))
        tensor_prod = TensorProduct(*self.args)
        return mul * Add(*[tensor_prod(*p[0]) * p[1] for p in zip(perms, perms_par)])
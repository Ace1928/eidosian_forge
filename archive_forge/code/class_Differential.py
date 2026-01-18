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
class Differential(Expr):
    """Return the differential (exterior derivative) of a form field.

    Explanation
    ===========

    The differential of a form (i.e. the exterior derivative) has a complicated
    definition in the general case.
    The differential `df` of the 0-form `f` is defined for any vector field `v`
    as `df(v) = v(f)`.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import Differential
    >>> from sympy import pprint

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> g = Function('g')
    >>> s_field = g(fx, fy)
    >>> dg = Differential(s_field)

    >>> dg
    d(g(x, y))
    >>> pprint(dg(e_x))
    / d           \\|
    |---(g(xi, y))||
    \\dxi          /|xi=x
    >>> pprint(dg(e_y))
    / d           \\|
    |---(g(x, xi))||
    \\dxi          /|xi=y

    Applying the exterior derivative operator twice always results in:

    >>> Differential(dg)
    0
    """
    is_commutative = False

    def __new__(cls, form_field):
        if contravariant_order(form_field):
            raise ValueError('A vector field was supplied as an argument to Differential.')
        if isinstance(form_field, Differential):
            return S.Zero
        else:
            obj = super().__new__(cls, form_field)
            obj._form_field = form_field
            return obj

    @property
    def form_field(self):
        return self.args[0]

    def __call__(self, *vector_fields):
        """Apply on a list of vector_fields.

        Explanation
        ===========

        If the number of vector fields supplied is not equal to 1 + the order of
        the form field inside the differential the result is undefined.

        For 1-forms (i.e. differentials of scalar fields) the evaluation is
        done as `df(v)=v(f)`. However if `v` is ``None`` instead of a vector
        field, the differential is returned unchanged. This is done in order to
        permit partial contractions for higher forms.

        In the general case the evaluation is done by applying the form field
        inside the differential on a list with one less elements than the number
        of elements in the original list. Lowering the number of vector fields
        is achieved through replacing each pair of fields by their
        commutator.

        If the arguments are not vectors or ``None``s an error is raised.
        """
        if any(((contravariant_order(a) != 1 or covariant_order(a)) and a is not None for a in vector_fields)):
            raise ValueError('The arguments supplied to Differential should be vector fields or Nones.')
        k = len(vector_fields)
        if k == 1:
            if vector_fields[0]:
                return vector_fields[0].rcall(self._form_field)
            return self
        else:
            f = self._form_field
            v = vector_fields
            ret = 0
            for i in range(k):
                t = v[i].rcall(f.rcall(*v[:i] + v[i + 1:]))
                ret += (-1) ** i * t
                for j in range(i + 1, k):
                    c = Commutator(v[i], v[j])
                    if c:
                        t = f.rcall(*(c,) + v[:i] + v[i + 1:j] + v[j + 1:])
                        ret += (-1) ** (i + j) * t
            return ret
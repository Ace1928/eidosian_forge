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
class BaseCovarDerivativeOp(Expr):
    """Covariant derivative operator with respect to a base vector.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import BaseCovarDerivativeOp
    >>> from sympy.diffgeom import metric_to_Christoffel_2nd, TensorProduct

    >>> TP = TensorProduct
    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> ch = metric_to_Christoffel_2nd(TP(dx, dx) + TP(dy, dy))
    >>> ch
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> cvd = BaseCovarDerivativeOp(R2_r, 0, ch)
    >>> cvd(fx)
    1
    >>> cvd(fx*e_x)
    e_x
    """

    def __new__(cls, coord_sys, index, christoffel):
        index = _sympify(index)
        christoffel = ImmutableDenseNDimArray(christoffel)
        obj = super().__new__(cls, coord_sys, index, christoffel)
        obj._coord_sys = coord_sys
        obj._index = index
        obj._christoffel = christoffel
        return obj

    @property
    def coord_sys(self):
        return self.args[0]

    @property
    def index(self):
        return self.args[1]

    @property
    def christoffel(self):
        return self.args[2]

    def __call__(self, field):
        """Apply on a scalar field.

        The action of a vector field on a scalar field is a directional
        differentiation.
        If the argument is not a scalar field the behaviour is undefined.
        """
        if covariant_order(field) != 0:
            raise NotImplementedError()
        field = vectors_in_basis(field, self._coord_sys)
        wrt_vector = self._coord_sys.base_vector(self._index)
        wrt_scalar = self._coord_sys.coord_function(self._index)
        vectors = list(field.atoms(BaseVectorField))
        d_funcs = [Function('_#_%s' % i)(wrt_scalar) for i, b in enumerate(vectors)]
        d_result = field.subs(list(zip(vectors, d_funcs)))
        d_result = wrt_vector(d_result)
        d_result = d_result.subs(list(zip(d_funcs, vectors)))
        derivs = []
        for v in vectors:
            d = Add(*[self._christoffel[k, wrt_vector._index, v._index] * v._coord_sys.base_vector(k) for k in range(v._coord_sys.dim)])
            derivs.append(d)
        to_subs = [wrt_vector(d) for d in d_funcs]
        result = d_result.subs(list(zip(to_subs, derivs)))
        result = result.subs(list(zip(d_funcs, vectors)))
        return result.doit()
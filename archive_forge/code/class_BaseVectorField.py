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
class BaseVectorField(Expr):
    """Base vector field over a manifold for a given coordinate system.

    Explanation
    ===========

    A vector field is an operator taking a scalar field and returning a
    directional derivative (which is also a scalar field).
    A base vector field is the same type of operator, however the derivation is
    specifically done with respect to a chosen coordinate.

    To define a base vector field you need to choose the coordinate system and
    the index of the coordinate.

    The use of the vector field after its definition is independent of the
    coordinate system in which it was defined, however due to limitations in the
    simplification routines you may arrive at more complicated expression if you
    use unappropriate coordinate systems.

    Parameters
    ==========
    coord_sys : CoordSystem

    index : integer

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.diffgeom.rn import R2_p, R2_r
    >>> from sympy.diffgeom import BaseVectorField
    >>> from sympy import pprint

    >>> x, y = R2_r.symbols
    >>> rho, theta = R2_p.symbols
    >>> fx, fy = R2_r.base_scalars()
    >>> point_p = R2_p.point([rho, theta])
    >>> point_r = R2_r.point([x, y])

    >>> g = Function('g')
    >>> s_field = g(fx, fy)

    >>> v = BaseVectorField(R2_r, 1)
    >>> pprint(v(s_field))
    / d           \\|
    |---(g(x, xi))||
    \\dxi          /|xi=y
    >>> pprint(v(s_field).rcall(point_r).doit())
    d
    --(g(x, y))
    dy
    >>> pprint(v(s_field).rcall(point_p))
    / d                        \\|
    |---(g(rho*cos(theta), xi))||
    \\dxi                       /|xi=rho*sin(theta)

    """
    is_commutative = False

    def __new__(cls, coord_sys, index, **kwargs):
        index = _sympify(index)
        obj = super().__new__(cls, coord_sys, index)
        obj._coord_sys = coord_sys
        obj._index = index
        return obj

    @property
    def coord_sys(self):
        return self.args[0]

    @property
    def index(self):
        return self.args[1]

    @property
    def patch(self):
        return self.coord_sys.patch

    @property
    def manifold(self):
        return self.coord_sys.manifold

    @property
    def dim(self):
        return self.manifold.dim

    def __call__(self, scalar_field):
        """Apply on a scalar field.
        The action of a vector field on a scalar field is a directional
        differentiation.
        If the argument is not a scalar field an error is raised.
        """
        if covariant_order(scalar_field) or contravariant_order(scalar_field):
            raise ValueError('Only scalar fields can be supplied as arguments to vector fields.')
        if scalar_field is None:
            return self
        base_scalars = list(scalar_field.atoms(BaseScalarField))
        d_var = self._coord_sys._dummy
        d_funcs = [Function('_#_%s' % i)(d_var) for i, b in enumerate(base_scalars)]
        d_result = scalar_field.subs(list(zip(base_scalars, d_funcs)))
        d_result = d_result.diff(d_var)
        coords = self._coord_sys.symbols
        d_funcs_deriv = [f.diff(d_var) for f in d_funcs]
        d_funcs_deriv_sub = []
        for b in base_scalars:
            jac = self._coord_sys.jacobian(b._coord_sys, coords)
            d_funcs_deriv_sub.append(jac[b._index, self._index])
        d_result = d_result.subs(list(zip(d_funcs_deriv, d_funcs_deriv_sub)))
        result = d_result.subs(list(zip(d_funcs, base_scalars)))
        result = result.subs(list(zip(coords, self._coord_sys.coord_functions())))
        return result.doit()
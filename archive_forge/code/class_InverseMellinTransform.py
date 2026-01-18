from functools import reduce, wraps
from itertools import repeat
from sympy.core import S, pi
from sympy.core.add import Add
from sympy.core.function import (
from sympy.core.mul import Mul
from sympy.core.numbers import igcd, ilcm
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.core.traversal import postorder_traversal
from sympy.functions.combinatorial.factorials import factorial, rf
from sympy.functions.elementary.complexes import re, arg, Abs
from sympy.functions.elementary.exponential import exp, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, tanh
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.piecewise import piecewise_fold
from sympy.functions.elementary.trigonometric import cos, cot, sin, tan
from sympy.functions.special.bessel import besselj
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import meijerg
from sympy.integrals import integrate, Integral
from sympy.integrals.meijerint import _dummy
from sympy.logic.boolalg import to_cnf, conjuncts, disjuncts, Or, And
from sympy.polys.polyroots import roots
from sympy.polys.polytools import factor, Poly
from sympy.polys.rootoftools import CRootOf
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
import sympy.integrals.laplace as _laplace
class InverseMellinTransform(IntegralTransform):
    """
    Class representing unevaluated inverse Mellin transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute inverse Mellin transforms, see the
    :func:`inverse_mellin_transform` docstring.
    """
    _name = 'Inverse Mellin'
    _none_sentinel = Dummy('None')
    _c = Dummy('c')

    def __new__(cls, F, s, x, a, b, **opts):
        if a is None:
            a = InverseMellinTransform._none_sentinel
        if b is None:
            b = InverseMellinTransform._none_sentinel
        return IntegralTransform.__new__(cls, F, s, x, a, b, **opts)

    @property
    def fundamental_strip(self):
        a, b = (self.args[3], self.args[4])
        if a is InverseMellinTransform._none_sentinel:
            a = None
        if b is InverseMellinTransform._none_sentinel:
            b = None
        return (a, b)

    def _compute_transform(self, F, s, x, **hints):
        hints.pop('simplify', True)
        global _allowed
        if _allowed is None:
            _allowed = {exp, gamma, sin, cos, tan, cot, cosh, sinh, tanh, coth, factorial, rf}
        for f in postorder_traversal(F):
            if f.is_Function and f.has(s) and (f.func not in _allowed):
                raise IntegralTransformError('Inverse Mellin', F, 'Component %s not recognised.' % f)
        strip = self.fundamental_strip
        return _inverse_mellin_transform(F, s, x, strip, **hints)

    def _as_integral(self, F, s, x):
        c = self.__class__._c
        return Integral(F * x ** (-s), (s, c - S.ImaginaryUnit * S.Infinity, c + S.ImaginaryUnit * S.Infinity)) / (2 * S.Pi * S.ImaginaryUnit)
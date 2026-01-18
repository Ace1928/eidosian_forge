from sympy.core import S, pi, I
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import (
from sympy.core.mul import Mul, prod
from sympy.core.relational import _canonical, Ge, Gt, Lt, Unequality, Eq
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.functions.elementary.complexes import (
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, asinh
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin, atan
from sympy.functions.special.bessel import besseli, besselj, besselk, bessely
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.error_functions import erf, erfc, Ei
from sympy.functions.special.gamma_functions import digamma, gamma, lowergamma
from sympy.integrals import integrate, Integral
from sympy.integrals.transforms import (
from sympy.logic.boolalg import to_cnf, conjuncts, disjuncts, Or, And
from sympy.matrices.matrices import MatrixBase
from sympy.polys.matrices.linsolve import _lin_eq2dict
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.polys.rationaltools import together
from sympy.polys.rootoftools import RootSum
from sympy.utilities.exceptions import (
from sympy.utilities.misc import debug, debugf
class InverseLaplaceTransform(IntegralTransform):
    """
    Class representing unevaluated inverse Laplace transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute inverse Laplace transforms, see the
    :func:`inverse_laplace_transform` docstring.
    """
    _name = 'Inverse Laplace'
    _none_sentinel = Dummy('None')
    _c = Dummy('c')

    def __new__(cls, F, s, x, plane, **opts):
        if plane is None:
            plane = InverseLaplaceTransform._none_sentinel
        return IntegralTransform.__new__(cls, F, s, x, plane, **opts)

    @property
    def fundamental_plane(self):
        plane = self.args[3]
        if plane is InverseLaplaceTransform._none_sentinel:
            plane = None
        return plane

    def _compute_transform(self, F, s, t, **hints):
        return _inverse_laplace_transform_integration(F, s, t, self.fundamental_plane, **hints)

    def _as_integral(self, F, s, t):
        c = self.__class__._c
        return Integral(exp(s * t) * F, (s, c - S.ImaginaryUnit * S.Infinity, c + S.ImaginaryUnit * S.Infinity)) / (2 * S.Pi * S.ImaginaryUnit)

    def doit(self, **hints):
        """
        Try to evaluate the transform in closed form.

        Explanation
        ===========

        Standard hints are the following:
        - ``noconds``:  if True, do not return convergence conditions. The
        default setting is `True`.
        - ``simplify``: if True, it simplifies the final result. The
        default setting is `False`.
        """
        _noconds = hints.get('noconds', True)
        _simplify = hints.get('simplify', False)
        debugf('[ILT doit] (%s, %s, %s)', (self.function, self.function_variable, self.transform_variable))
        s_ = self.function_variable
        t_ = self.transform_variable
        fn = self.function
        plane = self.fundamental_plane
        r = _inverse_laplace_transform(fn, s_, t_, plane, simplify=_simplify)
        if _noconds:
            return r[0]
        else:
            return r
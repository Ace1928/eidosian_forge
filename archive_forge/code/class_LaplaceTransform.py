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
class LaplaceTransform(IntegralTransform):
    """
    Class representing unevaluated Laplace transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute Laplace transforms, see the :func:`laplace_transform`
    docstring.

    If this is called with ``.doit()``, it returns the Laplace transform as an
    expression. If it is called with ``.doit(noconds=False)``, it returns a
    tuple containing the same expression, a convergence plane, and conditions.
    """
    _name = 'Laplace'

    def _compute_transform(self, f, t, s, **hints):
        _simplify = hints.get('simplify', False)
        LT = _laplace_transform_integration(f, t, s, simplify=_simplify)
        return LT

    def _as_integral(self, f, t, s):
        return Integral(f * exp(-s * t), (t, S.Zero, S.Infinity))

    def _collapse_extra(self, extra):
        conds = []
        planes = []
        for plane, cond in extra:
            conds.append(cond)
            planes.append(plane)
        cond = And(*conds)
        plane = Max(*planes)
        if cond == S.false:
            raise IntegralTransformError('Laplace', None, 'No combined convergence.')
        return (plane, cond)

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
        debugf('[LT doit] (%s, %s, %s)', (self.function, self.function_variable, self.transform_variable))
        t_ = self.function_variable
        s_ = self.transform_variable
        fn = self.function
        r = _laplace_transform(fn, t_, s_, simplify=_simplify)
        if _noconds:
            return r[0]
        else:
            return r
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
def _laplace_rule_delta(f, t, s):
    """
    If this function finds a factor ``DiracDelta(b*t-a)``, it applies the
    masking property of the delta distribution. For example, if it gets
    ``(DiracDelta(t-a)*f(t), t, s)``, it will return
    ``(f(a)*exp(-a*s), -a, True)``.
    """
    a = Wild('a', exclude=[t])
    b = Wild('b', exclude=[t])
    y = Wild('y')
    z = Wild('z')
    ma1 = f.match(DiracDelta(y) * z)
    if ma1 and (not ma1[z].has(DiracDelta)):
        ma2 = ma1[y].collect(t).match(b * t - a)
        if ma2:
            debug('_laplace_apply_prog_rules match:')
            debugf('      f:    %s ( %s, %s )', (f, ma1, ma2))
            debug('      rule: multiply with DiracDelta')
            loc = ma2[a] / ma2[b]
            if re(loc) >= 0 and im(loc) == 0:
                r = exp(-ma2[a] / ma2[b] * s) * ma1[z].subs(t, ma2[a] / ma2[b]) / ma2[b]
                return (r, S.NegativeInfinity, S.true)
            else:
                return (0, S.NegativeInfinity, S.true)
        if ma1[y].is_polynomial(t):
            ro = roots(ma1[y], t)
            if roots is not {} and set(ro.values()) == {1}:
                slope = diff(ma1[y], t)
                r = Add(*[exp(-x * s) * ma1[z].subs(t, s) / slope.subs(t, x) for x in list(ro.keys()) if im(x) == 0 and re(x) >= 0])
                return (r, S.NegativeInfinity, S.true)
    return None
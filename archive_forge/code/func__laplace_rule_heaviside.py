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
def _laplace_rule_heaviside(f, t, s):
    """
    This function deals with time-shifted Heaviside step functions. If the time
    shift is positive, it applies the time-shift rule of the Laplace transform.
    For example, if it gets ``(Heaviside(t-a)*f(t), t, s)``, it will compute
    ``exp(-a*s)*LaplaceTransform(f(t+a), t, s)``.

    If the time shift is negative, the Heaviside function is simply removed
    as it means nothing to the Laplace transform.

    The function does not remove a factor ``Heaviside(t)``; this is done by
    the simple rules.
    """
    a = Wild('a', exclude=[t])
    y = Wild('y')
    g = Wild('g')
    ma1 = f.match(Heaviside(y) * g)
    if ma1:
        ma2 = ma1[y].match(t - a)
        if ma2 and ma2[a].is_positive:
            debug('_laplace_apply_prog_rules match:')
            debugf('      f:    %s ( %s, %s )', (f, ma1, ma2))
            debug('      rule: time shift (4.1.4)')
            r, pr, cr = _laplace_transform(ma1[g].subs(t, t + ma2[a]), t, s, simplify=False)
            return (exp(-ma2[a] * s) * r, pr, cr)
        if ma2 and ma2[a].is_negative:
            debug('_laplace_apply_prog_rules match:')
            debugf('      f:    %s ( %s, %s )', (f, ma1, ma2))
            debug('      rule: Heaviside factor, negative time shift (4.1.4)')
            r, pr, cr = _laplace_transform(ma1[g], t, s, simplify=False)
            return (r, pr, cr)
    return None
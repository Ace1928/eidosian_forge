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
def _laplace_rule_exp(f, t, s):
    """
    If this function finds a factor ``exp(a*t)``, it applies the
    frequency-shift rule of the Laplace transform and adjusts the convergence
    plane accordingly.  For example, if it gets ``(exp(-a*t)*f(t), t, s)``, it
    will compute ``LaplaceTransform(f(t), t, s+a)``.
    """
    a = Wild('a', exclude=[t])
    y = Wild('y')
    z = Wild('z')
    ma1 = f.match(exp(y) * z)
    if ma1:
        ma2 = ma1[y].collect(t).match(a * t)
        if ma2:
            debug('_laplace_apply_prog_rules match:')
            debugf('      f:    %s ( %s, %s )', (f, ma1, ma2))
            debug('      rule: multiply with exp (4.1.5)')
            r, pr, cr = _laplace_transform(ma1[z], t, s - ma2[a], simplify=False)
            return (r, pr + re(ma2[a]), cr)
    return None
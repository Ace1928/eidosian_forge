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
def _laplace_rule_diff(f, t, s, doit=True, **hints):
    """
    This function looks for derivatives in the time domain and replaces it
    by factors of `s` and initial conditions in the frequency domain. For
    example, if it gets ``(diff(f(t), t), t, s)``, it will compute
    ``s*LaplaceTransform(f(t), t, s) - f(0)``.
    """
    a = Wild('a', exclude=[t])
    n = Wild('n', exclude=[t])
    g = WildFunction('g')
    ma1 = f.match(a * Derivative(g, (t, n)))
    if ma1 and ma1[n].is_integer:
        m = [z.has(t) for z in ma1[g].args]
        if sum(m) == 1:
            debug('_laplace_apply_rules match:')
            debugf('      f, n: %s, %s', (f, ma1[n]))
            debug('      rule: time derivative (4.1.8)')
            d = []
            for k in range(ma1[n]):
                if k == 0:
                    y = ma1[g].subs(t, 0)
                else:
                    y = Derivative(ma1[g], (t, k)).subs(t, 0)
                d.append(s ** (ma1[n] - k - 1) * y)
            r, pr, cr = _laplace_transform(ma1[g], t, s, simplify=False)
            return (ma1[a] * (s ** ma1[n] * r - Add(*d)), pr, cr)
    return None
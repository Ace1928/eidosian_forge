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
def _inverse_laplace_time_diff(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    n = Wild('n', exclude=[s])
    g = Wild('g')
    ma1 = F.match(s ** n * g)
    if ma1 and ma1[n].is_integer and ma1[n].is_positive:
        debug('_inverse_laplace_time_diff match:')
        debugf('      f:    %s', (F,))
        debug('      rule: s**n*F(s) o---o diff(f(t), t, n)')
        debugf('      ma:   %s', (ma1,))
        r, c = _inverse_laplace_transform(ma1[g], s, t, plane)
        r = r.replace(Heaviside(t), 1)
        if r.has(InverseLaplaceTransform):
            return (diff(r, t, ma1[n]), c)
        else:
            return (Heaviside(t) * diff(r, t, ma1[n]), c)
    return None
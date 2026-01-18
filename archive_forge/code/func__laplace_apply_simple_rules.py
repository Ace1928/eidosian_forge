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
def _laplace_apply_simple_rules(f, t, s):
    """
    This function applies all simple rules and returns the result if one
    of them gives a result.
    """
    simple_rules, t_, s_ = _laplace_build_rules()
    prep_old = ''
    prep_f = ''
    for t_dom, s_dom, check, plane, prep in simple_rules:
        if prep_old != prep:
            prep_f = prep(f.subs({t: t_}))
            prep_old = prep
        ma = prep_f.match(t_dom)
        if ma:
            try:
                c = check.xreplace(ma)
            except TypeError:
                continue
            if c == S.true:
                debug('_laplace_apply_simple_rules match:')
                debugf('      f:     %s', (f,))
                debugf('      rule:  %s o---o %s', (t_dom, s_dom))
                debugf('      match: %s', (ma,))
                return (s_dom.xreplace(ma).subs({s_: s}), plane.xreplace(ma), S.true)
    return None
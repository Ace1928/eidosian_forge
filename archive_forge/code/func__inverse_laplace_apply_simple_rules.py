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
def _inverse_laplace_apply_simple_rules(f, s, t):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    if f == 1:
        debug('_inverse_laplace_apply_simple_rules match:')
        debugf('      f:    %s', (1,))
        debugf('      rule: 1 o---o DiracDelta(%s)', (t,))
        return (DiracDelta(t), S.true)
    _ILT_rules, s_, t_ = _inverse_laplace_build_rules()
    _prep = ''
    fsubs = f.subs({s: s_})
    for s_dom, t_dom, check, prep, fac in _ILT_rules:
        if _prep != (prep, fac):
            _F = prep(fsubs * fac)
            _prep = (prep, fac)
        ma = _F.match(s_dom)
        if ma:
            try:
                c = check.xreplace(ma)
            except TypeError:
                continue
            if c == S.true:
                debug('_inverse_laplace_apply_simple_rules match:')
                debugf('      f:    %s', (f,))
                debugf('      rule: %s o---o %s', (s_dom, t_dom))
                debugf('      ma:   %s', (ma,))
                return (Heaviside(t) * t_dom.xreplace(ma).subs({t_: t}), S.true)
    return None
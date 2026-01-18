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
def _laplace_transform(fn, t_, s_, simplify=True):
    """
    Front-end function of the Laplace transform. It tries to apply all known
    rules recursively, and if everything else fails, it tries to integrate.
    """
    debugf('[LT _l_t] (%s, %s, %s)', (fn, t_, s_))
    terms = Add.make_args(fn)
    terms_s = []
    planes = []
    conditions = []
    for ff in terms:
        k, ft = ff.as_independent(t_, as_Add=False)
        if (r := _laplace_apply_simple_rules(ft, t_, s_)) is not None:
            pass
        elif (r := _laplace_apply_prog_rules(ft, t_, s_)) is not None:
            pass
        elif (r := _laplace_expand(ft, t_, s_)) is not None:
            pass
        elif any((undef.has(t_) for undef in ft.atoms(AppliedUndef))):
            r = (LaplaceTransform(ft, t_, s_), S.NegativeInfinity, True)
        elif (r := _laplace_transform_integration(ft, t_, s_, simplify=simplify)) is not None:
            pass
        else:
            r = (LaplaceTransform(ft, t_, s_), S.NegativeInfinity, True)
        ri_, pi_, ci_ = r
        terms_s.append(k * ri_)
        planes.append(pi_)
        conditions.append(ci_)
    result = Add(*terms_s)
    if simplify:
        result = result.simplify(doit=False)
    plane = Max(*planes)
    condition = And(*conditions)
    return (result, plane, condition)
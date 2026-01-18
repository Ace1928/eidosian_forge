from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyclasses import ANP
from sympy.polys.polyerrors import CoercionFailed, DomainError, NotAlgebraic, IsomorphismFailed
from sympy.utilities import public
def _make_converter(K):
    """Construct the converter to convert back to Expr"""
    gen = K.ext.as_expr()
    todom = K.dom.from_sympy
    powers = [S.One, gen]
    for n in range(2, K.mod.degree()):
        powers.append((gen * powers[-1]).expand())
    terms = [dict((t.as_coeff_Mul()[::-1] for t in Add.make_args(p))) for p in powers]
    algebraics = set().union(*terms)
    matrix = [[todom(t.get(a, S.Zero)) for t in terms] for a in algebraics]

    def converter(a):
        """Convert a to Expr using converter"""
        ai = a.rep[::-1]
        tosympy = K.dom.to_sympy
        coeffs_dom = [sum((mij * aj for mij, aj in zip(mi, ai))) for mi in matrix]
        coeffs_sympy = [tosympy(c) for c in coeffs_dom]
        res = Add(*(Mul(c, a) for c, a in zip(coeffs_sympy, algebraics)))
        return res
    return converter
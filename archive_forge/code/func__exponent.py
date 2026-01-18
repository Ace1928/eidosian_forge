from __future__ import annotations
from itertools import permutations
from functools import reduce
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.symbol import Wild, Dummy, Symbol
from sympy.core.basic import sympify
from sympy.core.numbers import Rational, pi, I
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.traversal import iterfreeargs
from sympy.functions import exp, sin, cos, tan, cot, asin, atan
from sympy.functions import log, sinh, cosh, tanh, coth, asinh
from sympy.functions import sqrt, erf, erfi, li, Ei
from sympy.functions import besselj, bessely, besseli, besselk
from sympy.functions import hankel1, hankel2, jn, yn
from sympy.functions.elementary.complexes import Abs, re, im, sign, arg
from sympy.functions.elementary.exponential import LambertW
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.simplify.radsimp import collect
from sympy.logic.boolalg import And, Or
from sympy.utilities.iterables import uniq
from sympy.polys import quo, gcd, lcm, factor_list, cancel, PolynomialError
from sympy.polys.monomials import itermonomials
from sympy.polys.polyroots import root_factors
from sympy.polys.rings import PolyRing
from sympy.polys.solvers import solve_lin_sys
from sympy.polys.constructor import construct_domain
from sympy.integrals.integrals import integrate
def _exponent(g):
    if g.is_Pow:
        if g.exp.is_Rational and g.exp.q != 1:
            if g.exp.p > 0:
                return g.exp.p + g.exp.q - 1
            else:
                return abs(g.exp.p + g.exp.q)
        else:
            return 1
    elif not g.is_Atom and g.args:
        return max([_exponent(h) for h in g.args])
    else:
        return 1
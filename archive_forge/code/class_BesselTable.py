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
class BesselTable:
    """
    Derivatives of Bessel functions of orders n and n-1
    in terms of each other.

    See the docstring of DiffCache.
    """

    def __init__(self):
        self.table = {}
        self.n = Dummy('n')
        self.z = Dummy('z')
        self._create_table()

    def _create_table(t):
        table, n, z = (t.table, t.n, t.z)
        for f in (besselj, bessely, hankel1, hankel2):
            table[f] = (f(n - 1, z) - n * f(n, z) / z, (n - 1) * f(n - 1, z) / z - f(n, z))
        f = besseli
        table[f] = (f(n - 1, z) - n * f(n, z) / z, (n - 1) * f(n - 1, z) / z + f(n, z))
        f = besselk
        table[f] = (-f(n - 1, z) - n * f(n, z) / z, (n - 1) * f(n - 1, z) / z - f(n, z))
        for f in (jn, yn):
            table[f] = (f(n - 1, z) - (n + 1) * f(n, z) / z, (n - 1) * f(n - 1, z) / z - f(n, z))

    def diffs(t, f, n, z):
        if f in t.table:
            diff0, diff1 = t.table[f]
            repl = [(t.n, n), (t.z, z)]
            return (diff0.subs(repl), diff1.subs(repl))

    def has(t, f):
        return f in t.table
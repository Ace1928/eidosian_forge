from collections import defaultdict
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.exprtools import Factors, gcd_terms, factor_terms
from sympy.core.function import expand_mul
from sympy.core.mul import Mul
from sympy.core.numbers import pi, I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import bottom_up
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import (
from sympy.functions.elementary.trigonometric import (
from sympy.ntheory.factor_ import perfect_power
from sympy.polys.polytools import factor
from sympy.strategies.tree import greedy
from sympy.strategies.core import identity, debug
from sympy import SYMPY_DEBUG
def _TR11(rv):
    """
    Helper for TR11 to find half-arguments for sin in factors of
    num/den that appear in cos or sin factors in the den/num.

    Examples
    ========

    >>> from sympy.simplify.fu import TR11, _TR11
    >>> from sympy import cos, sin
    >>> from sympy.abc import x
    >>> TR11(sin(x/3)/(cos(x/6)))
    sin(x/3)/cos(x/6)
    >>> _TR11(sin(x/3)/(cos(x/6)))
    2*sin(x/6)
    >>> TR11(sin(x/6)/(sin(x/3)))
    sin(x/6)/sin(x/3)
    >>> _TR11(sin(x/6)/(sin(x/3)))
    1/(2*cos(x/6))

    """

    def f(rv):
        if not isinstance(rv, Expr):
            return rv

        def sincos_args(flat):
            args = defaultdict(set)
            for fi in Mul.make_args(flat):
                b, e = fi.as_base_exp()
                if e.is_Integer and e > 0:
                    if b.func in (cos, sin):
                        args[type(b)].add(b.args[0])
            return args
        num_args, den_args = map(sincos_args, rv.as_numer_denom())

        def handle_match(rv, num_args, den_args):
            for narg in num_args[sin]:
                half = narg / 2
                if half in den_args[cos]:
                    func = cos
                elif half in den_args[sin]:
                    func = sin
                else:
                    continue
                rv = TR11(rv, half)
                den_args[func].remove(half)
            return rv
        rv = handle_match(rv, num_args, den_args)
        rv = handle_match(rv, den_args, num_args)
        return rv
    return bottom_up(rv, f)
from collections import defaultdict
from functools import reduce
from sympy.core import (sympify, Basic, S, Expr, factor_terms,
from sympy.core.cache import cacheit
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
from sympy.core.numbers import I, Integer, igcd
from sympy.core.sorting import _nodes
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.functions import atan2
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.polys import Poly, factor, cancel, parallel_poly_from_expr
from sympy.polys.domains import ZZ
from sympy.polys.polyerrors import PolificationFailed
from sympy.polys.polytools import groebner
from sympy.simplify.cse_main import cse
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
def _futrig(e):
    """Helper for futrig."""
    from sympy.simplify.fu import TR1, TR2, TR3, TR2i, TR10, L, TR10i, TR8, TR6, TR15, TR16, TR111, TR5, TRmorrie, TR11, _TR11, TR14, TR22, TR12
    if not e.has(TrigonometricFunction):
        return e
    if e.is_Mul:
        coeff, e = e.as_independent(TrigonometricFunction)
    else:
        coeff = None
    Lops = lambda x: (L(x), x.count_ops(), _nodes(x), len(x.args), x.is_Add)
    trigs = lambda x: x.has(TrigonometricFunction)
    tree = [identity, (TR3, TR1, TR12, lambda x: _eapply(factor, x, trigs), TR2, [identity, lambda x: _eapply(_mexpand, x, trigs)], TR2i, lambda x: _eapply(lambda i: factor(i.normal()), x, trigs), TR14, TR5, TR10, TR11, _TR11, TR6, lambda x: _eapply(factor, x, trigs), TR14, [identity, lambda x: _eapply(_mexpand, x, trigs)], TR10i, TRmorrie, [identity, TR8], [identity, lambda x: TR2i(TR2(x))], [lambda x: _eapply(expand_mul, TR5(x), trigs), lambda x: _eapply(expand_mul, TR15(x), trigs)], [lambda x: _eapply(expand_mul, TR6(x), trigs), lambda x: _eapply(expand_mul, TR16(x), trigs)], TR111, [identity, TR2i], [identity, lambda x: _eapply(expand_mul, TR22(x), trigs)], TR1, TR2, TR2i, [identity, lambda x: _eapply(factor_terms, TR12(x), trigs)])]
    e = greedy(tree, objective=Lops)(e)
    if coeff is not None:
        e = coeff * e
    return e
from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent
@cacheit
def _simplify_patterns_or():
    """ Two-term patterns for Or."""
    from sympy.core import Wild
    from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.miscellaneous import Min, Max
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    _matchers_or = ((Tuple(Le(b, a), Le(a, b)), true), (Tuple(Le(b, a), Ne(a, b)), true), (Tuple(Eq(a, b), Le(a, b)), Le(a, b)), (Tuple(Eq(a, b), Lt(a, b)), Le(a, b)), (Tuple(Lt(b, a), Lt(a, b)), Ne(a, b)), (Tuple(Lt(b, a), Ne(a, b)), Ne(a, b)), (Tuple(Le(a, b), Lt(a, b)), Le(a, b)), (Tuple(Eq(a, b), Ne(a, c)), ITE(Eq(b, c), true, Ne(a, c))), (Tuple(Ne(a, b), Ne(a, c)), ITE(Eq(b, c), Ne(a, b), true)), (Tuple(Le(b, a), Le(c, a)), Ge(a, Min(b, c))), (Tuple(Le(b, a), Lt(c, a)), ITE(b > c, Lt(c, a), Le(b, a))), (Tuple(Lt(b, a), Lt(c, a)), Gt(a, Min(b, c))), (Tuple(Le(a, b), Le(a, c)), Le(a, Max(b, c))), (Tuple(Le(a, b), Lt(a, c)), ITE(b >= c, Le(a, b), Lt(a, c))), (Tuple(Lt(a, b), Lt(a, c)), Lt(a, Max(b, c))), (Tuple(Le(a, b), Le(c, a)), ITE(b >= c, true, Or(Le(a, b), Ge(a, c)))), (Tuple(Le(c, a), Le(a, b)), ITE(b >= c, true, Or(Le(a, b), Ge(a, c)))), (Tuple(Lt(a, b), Lt(c, a)), ITE(b > c, true, Or(Lt(a, b), Gt(a, c)))), (Tuple(Lt(c, a), Lt(a, b)), ITE(b > c, true, Or(Lt(a, b), Gt(a, c)))), (Tuple(Le(a, b), Lt(c, a)), ITE(b >= c, true, Or(Le(a, b), Gt(a, c)))), (Tuple(Le(c, a), Lt(a, b)), ITE(b >= c, true, Or(Lt(a, b), Ge(a, c)))), (Tuple(Lt(b, a), Lt(a, -b)), ITE(b >= 0, Gt(Abs(a), b), true)), (Tuple(Le(b, a), Le(a, -b)), ITE(b > 0, Ge(Abs(a), b), true)))
    return _matchers_or
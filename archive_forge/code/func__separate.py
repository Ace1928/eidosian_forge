from functools import reduce
from itertools import combinations_with_replacement
from sympy.simplify import simplify  # type: ignore
from sympy.core import Add, S
from sympy.core.function import Function, expand, AppliedUndef, Subs
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, symbols
from sympy.functions import exp
from sympy.integrals.integrals import Integral, integrate
from sympy.utilities.iterables import has_dups, is_sequence
from sympy.utilities.misc import filldedent
from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from sympy.solvers.solvers import solve
from sympy.simplify.radsimp import collect
import operator
def _separate(eq, dep, others):
    """Separate expression into two parts based on dependencies of variables."""
    terms = set()
    for term in eq.args:
        if term.is_Mul:
            for i in term.args:
                if i.is_Derivative and (not i.has(*others)):
                    terms.add(term)
                    continue
        elif term.is_Derivative and (not term.has(*others)):
            terms.add(term)
    div = set()
    for term in terms:
        ext, sep = term.expand().as_independent(dep)
        if sep.has(*others):
            return None
        div.add(ext)
    if len(div) > 0:
        eq = Add(*[simplify(Add(*[term / i for i in div])) for term in eq.args])
    div = set()
    lhs = rhs = 0
    for term in eq.args:
        if not term.has(*others):
            lhs += term
            continue
        temp, sep = term.expand().as_independent(dep)
        if sep.has(*others):
            return None
        div.add(sep)
        rhs -= term.expand()
    fulldiv = reduce(operator.add, div)
    lhs = simplify(lhs / fulldiv).expand()
    rhs = simplify(rhs / fulldiv).expand()
    if lhs.has(*others) or rhs.has(dep):
        return None
    return [lhs, rhs]
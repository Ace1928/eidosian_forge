from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
def _determine_ambiguous(term, ordered, ambiguous_groups):
    all_ambiguous = set()
    for dummies in ambiguous_groups:
        all_ambiguous |= dummies
    all_ordered = set(ordered) - all_ambiguous
    if not all_ordered:
        group = [d for d in ordered if d in ambiguous_groups[0]]
        d = group[0]
        all_ordered.add(d)
        ambiguous_groups[0].remove(d)
    stored_counter = _symbol_factory._counter
    subslist = []
    for d in [d for d in ordered if d in all_ordered]:
        nondum = _symbol_factory._next()
        subslist.append((d, nondum))
    newterm = term.subs(subslist)
    neworder = _get_ordered_dummies(newterm)
    _symbol_factory._set_counter(stored_counter)
    for group in ambiguous_groups:
        ordered_group = [d for d in neworder if d in group]
        ordered_group.reverse()
        result = []
        for d in ordered:
            if d in group:
                result.append(ordered_group.pop())
            else:
                result.append(d)
        ordered = result
    return ordered
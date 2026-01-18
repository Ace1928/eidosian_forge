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
def _apply_patternbased_twoterm_simplification(Rel, patterns, func, dominatingvalue, replacementvalue, measure):
    """ Apply pattern-based two-term simplification."""
    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy.core.relational import Ge, Gt, _Inequality
    changed = True
    while changed and len(Rel) >= 2:
        changed = False
        Rel = [r.reversed if isinstance(r, (Ge, Gt)) else r for r in Rel]
        Rel = list(ordered(Rel))
        rtmp = [(r,) if isinstance(r, _Inequality) else (r, r.reversed) for r in Rel]
        results = []
        for (i, pi), (j, pj) in combinations(enumerate(rtmp), 2):
            for pattern, simp in patterns:
                res = []
                for p1, p2 in product(pi, pj):
                    oldexpr = Tuple(p1, p2)
                    tmpres = oldexpr.match(pattern)
                    if tmpres:
                        res.append((tmpres, oldexpr))
                if res:
                    for tmpres, oldexpr in res:
                        np = simp.xreplace(tmpres)
                        if np == dominatingvalue:
                            return [replacementvalue]
                        if not isinstance(np, ITE) and (not np.has(Min, Max)):
                            costsaving = measure(func(*oldexpr.args)) - measure(np)
                            if costsaving > 0:
                                results.append((costsaving, ([i, j], np)))
        if results:
            results = sorted(results, key=lambda pair: pair[0], reverse=True)
            replacement = results[0][1]
            idx, newrel = replacement
            idx.sort()
            for index in reversed(idx):
                del Rel[index]
            if dominatingvalue is None or newrel != Not(dominatingvalue):
                if newrel.func == func:
                    for a in newrel.args:
                        Rel.append(a)
                else:
                    Rel.append(newrel)
            changed = True
    return Rel
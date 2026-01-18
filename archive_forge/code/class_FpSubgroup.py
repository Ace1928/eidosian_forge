from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.combinatorics.free_groups import (FreeGroup, FreeGroupElement,
from sympy.combinatorics.rewritingsystem import RewritingSystem
from sympy.combinatorics.coset_table import (CosetTable,
from sympy.combinatorics import PermutationGroup
from sympy.matrices.normalforms import invariant_factors
from sympy.matrices import Matrix
from sympy.polys.polytools import gcd
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.magic import pollute
from itertools import product
class FpSubgroup(DefaultPrinting):
    """
    The class implementing a subgroup of an FpGroup or a FreeGroup
    (only finite index subgroups are supported at this point). This
    is to be used if one wishes to check if an element of the original
    group belongs to the subgroup

    """

    def __init__(self, G, gens, normal=False):
        super().__init__()
        self.parent = G
        self.generators = list({g for g in gens if g != G.identity})
        self._min_words = None
        self.C = None
        self.normal = normal

    def __contains__(self, g):
        if isinstance(self.parent, FreeGroup):
            if self._min_words is None:

                def _process(w):
                    p, r = w.cyclic_reduction(removed=True)
                    if not r.is_identity:
                        return [(r, p)]
                    else:
                        return [w, w ** (-1)]
                gens = []
                for w in self.generators:
                    if self.normal:
                        w = w.cyclic_reduction()
                    gens.extend(_process(w))
                for w1 in gens:
                    for w2 in gens:
                        if w1 == w2 or (not isinstance(w1, tuple) and w1 ** (-1) == w2):
                            continue
                        if isinstance(w1, tuple):
                            s1, s2 = (w1[0][0], w1[0][0] ** (-1))
                        else:
                            s1, s2 = (w1[0], w1[len(w1) - 1])
                        if isinstance(w2, tuple):
                            r1, r2 = (w2[0][0], w2[0][0] ** (-1))
                        else:
                            r1, r2 = (w2[0], w2[len(w1) - 1])
                        p1, p2 = (w1, w2)
                        if isinstance(w1, tuple):
                            p1 = w1[0] * w1[1] * w1[0] ** (-1)
                        if isinstance(w2, tuple):
                            p2 = w2[0] * w2[1] * w2[0] ** (-1)
                        if r1 ** (-1) == s2 and (not (p1 * p2).is_identity):
                            new = _process(p1 * p2)
                            if new not in gens:
                                gens.extend(new)
                        if r2 ** (-1) == s1 and (not (p2 * p1).is_identity):
                            new = _process(p2 * p1)
                            if new not in gens:
                                gens.extend(new)
                self._min_words = gens
            min_words = self._min_words

            def _is_subword(w):
                w, r = w.cyclic_reduction(removed=True)
                if r.is_identity or self.normal:
                    return w in min_words
                else:
                    t = [s[1] for s in min_words if isinstance(s, tuple) and s[0] == r]
                    return [s for s in t if w.power_of(s)] != []
            known = {}

            def _word_break(w):
                if len(w) == 0:
                    return True
                i = 0
                while i < len(w):
                    i += 1
                    prefix = w.subword(0, i)
                    if not _is_subword(prefix):
                        continue
                    rest = w.subword(i, len(w))
                    if rest not in known:
                        known[rest] = _word_break(rest)
                    if known[rest]:
                        return True
                return False
            if self.normal:
                g = g.cyclic_reduction()
            return _word_break(g)
        else:
            if self.C is None:
                C = self.parent.coset_enumeration(self.generators)
                self.C = C
            i = 0
            C = self.C
            for j in range(len(g)):
                i = C.table[i][C.A_dict[g[j]]]
            return i == 0

    def order(self):
        if not self.generators:
            return S.One
        if isinstance(self.parent, FreeGroup):
            return S.Infinity
        if self.C is None:
            C = self.parent.coset_enumeration(self.generators)
            self.C = C
        return self.parent.order() / len(self.C.table)

    def to_FpGroup(self):
        if isinstance(self.parent, FreeGroup):
            gen_syms = ['x_%d' % i for i in range(len(self.generators))]
            return free_group(', '.join(gen_syms))[0]
        return self.parent.subgroup(C=self.C)

    def __str__(self):
        if len(self.generators) > 30:
            str_form = '<fp subgroup with %s generators>' % len(self.generators)
        else:
            str_form = '<fp subgroup on the generators %s>' % str(self.generators)
        return str_form
    __repr__ = __str__
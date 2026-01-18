from math import factorial as _factorial, log, prod
from itertools import chain, islice, product
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import (_af_commutes_with, _af_invert,
from sympy.combinatorics.util import (_check_cycles_alt_sym,
from sympy.core import Basic
from sympy.core.random import _randrange, randrange, choice
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import primefactors, sieve
from sympy.ntheory.factor_ import (factorint, multiplicity)
from sympy.ntheory.primetest import isprime
from sympy.utilities.iterables import has_variety, is_sequence, uniq
def _orbit_transversal(degree, generators, alpha, pairs, af=False, slp=False):
    """Computes a transversal for the orbit of ``alpha`` as a set.

    Explanation
    ===========

    generators   generators of the group ``G``

    For a permutation group ``G``, a transversal for the orbit
    `Orb = \\{g(\\alpha) | g \\in G\\}` is a set
    `\\{g_\\beta | g_\\beta(\\alpha) = \\beta\\}` for `\\beta \\in Orb`.
    Note that there may be more than one possible transversal.
    If ``pairs`` is set to ``True``, it returns the list of pairs
    `(\\beta, g_\\beta)`. For a proof of correctness, see [1], p.79

    if ``af`` is ``True``, the transversal elements are given in
    array form.

    If `slp` is `True`, a dictionary `{beta: slp_beta}` is returned
    for `\\beta \\in Orb` where `slp_beta` is a list of indices of the
    generators in `generators` s.t. if `slp_beta = [i_1 \\dots i_n]`
    `g_\\beta = generators[i_n] \\times \\dots \\times generators[i_1]`.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> from sympy.combinatorics.perm_groups import _orbit_transversal
    >>> G = DihedralGroup(6)
    >>> _orbit_transversal(G.degree, G.generators, 0, False)
    [(5), (0 1 2 3 4 5), (0 5)(1 4)(2 3), (0 2 4)(1 3 5), (5)(0 4)(1 3), (0 3)(1 4)(2 5)]
    """
    tr = [(alpha, list(range(degree)))]
    slp_dict = {alpha: []}
    used = [False] * degree
    used[alpha] = True
    gens = [x._array_form for x in generators]
    for x, px in tr:
        px_slp = slp_dict[x]
        for gen in gens:
            temp = gen[x]
            if used[temp] == False:
                slp_dict[temp] = [gens.index(gen)] + px_slp
                tr.append((temp, _af_rmul(gen, px)))
                used[temp] = True
    if pairs:
        if not af:
            tr = [(x, _af_new(y)) for x, y in tr]
        if not slp:
            return tr
        return (tr, slp_dict)
    if af:
        tr = [y for _, y in tr]
        if not slp:
            return tr
        return (tr, slp_dict)
    tr = [_af_new(y) for _, y in tr]
    if not slp:
        return tr
    return (tr, slp_dict)
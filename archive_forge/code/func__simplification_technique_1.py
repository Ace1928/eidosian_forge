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
def _simplification_technique_1(rels):
    """
    All relators are checked to see if they are of the form `gen^n`. If any
    such relators are found then all other relators are processed for strings
    in the `gen` known order.

    Examples
    ========

    >>> from sympy.combinatorics import free_group
    >>> from sympy.combinatorics.fp_groups import _simplification_technique_1
    >>> F, x, y = free_group("x, y")
    >>> w1 = [x**2*y**4, x**3]
    >>> _simplification_technique_1(w1)
    [x**-1*y**4, x**3]

    >>> w2 = [x**2*y**-4*x**5, x**3, x**2*y**8, y**5]
    >>> _simplification_technique_1(w2)
    [x**-1*y*x**-1, x**3, x**-1*y**-2, y**5]

    >>> w3 = [x**6*y**4, x**4]
    >>> _simplification_technique_1(w3)
    [x**2*y**4, x**4]

    """
    rels = rels[:]
    exps = {}
    for i in range(len(rels)):
        rel = rels[i]
        if rel.number_syllables() == 1:
            g = rel[0]
            exp = abs(rel.array_form[0][1])
            if rel.array_form[0][1] < 0:
                rels[i] = rels[i] ** (-1)
                g = g ** (-1)
            if g in exps:
                exp = gcd(exp, exps[g].array_form[0][1])
            exps[g] = g ** exp
    one_syllables_words = exps.values()
    for i in range(len(rels)):
        rel = rels[i]
        if rel in one_syllables_words:
            continue
        rel = rel.eliminate_words(one_syllables_words, _all=True)
        for g in rel.contains_generators():
            if g in exps:
                exp = exps[g].array_form[0][1]
                max_exp = (exp + 1) // 2
                rel = rel.eliminate_word(g ** max_exp, g ** (max_exp - exp), _all=True)
                rel = rel.eliminate_word(g ** (-max_exp), g ** (-(max_exp - exp)), _all=True)
        rels[i] = rel
    rels = [r.identity_cyclic_reduction() for r in rels]
    return rels
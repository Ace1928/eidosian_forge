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
def _simplify_relators(rels, identity):
    """Relies upon ``_simplification_technique_1`` for its functioning. """
    rels = rels[:]
    rels = list(set(_simplification_technique_1(rels)))
    rels.sort()
    rels = [r.identity_cyclic_reduction() for r in rels]
    try:
        rels.remove(identity)
    except ValueError:
        pass
    return rels
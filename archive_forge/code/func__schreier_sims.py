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
def _schreier_sims(self, base=None):
    schreier = self.schreier_sims_incremental(base=base, slp_dict=True)
    base, strong_gens = schreier[:2]
    self._base = base
    self._strong_gens = strong_gens
    self._strong_gens_slp = schreier[2]
    if not base:
        self._transversals = []
        self._basic_orbits = []
        return
    strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    basic_orbits, transversals, slps = _orbits_transversals_from_bsgs(base, strong_gens_distr, slp=True)
    for i, slp in enumerate(slps):
        gens = strong_gens_distr[i]
        for k in slp:
            slp[k] = [strong_gens.index(gens[s]) for s in slp[k]]
    self._transversals = transversals
    self._basic_orbits = [sorted(x) for x in basic_orbits]
    self._transversal_slp = slps
from __future__ import print_function
from patsy.util import no_pickling
def _simplify_one_subterm(subterms):
    for short_i, short_subterm in enumerate(subterms):
        for long_i, long_subterm in enumerate(subterms[short_i + 1:]):
            if long_subterm.can_absorb(short_subterm):
                new_subterm = long_subterm.absorb(short_subterm)
                subterms[short_i + 1 + long_i] = new_subterm
                subterms.pop(short_i)
                return True
    return False
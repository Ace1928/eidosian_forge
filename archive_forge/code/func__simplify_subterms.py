from __future__ import print_function
from patsy.util import no_pickling
def _simplify_subterms(subterms):
    while _simplify_one_subterm(subterms):
        pass
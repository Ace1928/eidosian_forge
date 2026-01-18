from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def _filtered_terms(self, sgn):
    return [(p, e) for p, e in self._polymod_exponent_pairs if sgn * e >= 0]
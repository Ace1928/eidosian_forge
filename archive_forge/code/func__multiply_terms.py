from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def _multiply_terms(self, sgn):
    return prod([p ** abs(e) for p, e in self._filtered_terms(sgn)])
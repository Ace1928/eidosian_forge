from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def _numerator_terms(self):
    return self._filtered_terms(+1)
from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def _is_zero(self):
    return any((p == 0 and e > 0 for p, e in self._polymod_exponent_pairs))
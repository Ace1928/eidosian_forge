from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _L_function(zpq_triple, evenN=2):
    z, p, q = zpq_triple
    z = _convert_to_pari_float(z)
    p = _convert_to_pari_float(p)
    q = _convert_to_pari_float(q)
    f = pari('2 * Pi * I') / evenN
    Pi2 = pari('Pi * Pi')
    return _dilog(z) + (z.log() + p * f) * ((1 - z).log() + q * f) / 2 - Pi2 / 6
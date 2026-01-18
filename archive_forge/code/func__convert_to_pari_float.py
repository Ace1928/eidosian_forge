from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _convert_to_pari_float(z):
    if type(z) == Gen and z.type() in ['t_INT', 't_FRAC']:
        return z * pari('1.0')
    return pari(z)
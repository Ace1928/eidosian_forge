from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _compute_flattening(a, b, c, d, branch_factor, N=2):
    PiMinusEpsilon = pari(3.141592)

    def safe_log(z):
        l = (branch_factor * z ** N).log()
        if l.imag().abs() > PiMinusEpsilon:
            raise LogToCloseToBranchCutError()
        return l / N
    a = _convert_to_pari_float(a)
    b = _convert_to_pari_float(b)
    c = _convert_to_pari_float(c)
    d = _convert_to_pari_float(d)
    w = safe_log(a) + safe_log(b) - safe_log(c) - safe_log(d)
    return w
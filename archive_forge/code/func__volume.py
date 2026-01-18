from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _volume(z):
    z = _convert_to_pari_float(z)
    return (1 - z).arg() * z.abs().log() + _dilog(z).imag()
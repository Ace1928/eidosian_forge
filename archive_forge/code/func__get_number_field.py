from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _get_number_field(d):
    for value in d.values():
        if isinstance(value, RUR):
            nf = value.number_field()
            if nf:
                return nf
        if type(value) == Gen and value.type() == 't_POLMOD':
            return value.mod()
    return None
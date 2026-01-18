from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _N_for_shapes(solution_dict):

    def get_N(key):
        m = re.match('zp{0,2}_(\\d{4})_\\d+$', key)
        if not m:
            raise Exception("Not a valid shape key: '%s'" % key)
        return sum([int(char) for char in m.group(1)]) + 2
    l = [get_N(key) for key in solution_dict.keys()]
    if not len(set(l)) == 1:
        raise Exception('Shape keys for different N')
    return l[0]
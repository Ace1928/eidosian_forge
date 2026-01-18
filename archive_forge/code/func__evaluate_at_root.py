from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _evaluate_at_root(p, root):
    if type(p) == Gen and p.type() == 't_POLMOD':
        return p.lift().substpol('x', root)
    if isinstance(p, RUR):
        return p.evaluate_at_root(root)
    return p
from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _apply_to_RURs(d, RUR_method):

    def _apply_to_RUR(v):
        if isinstance(v, RUR):
            return RUR_method(v)
        return v
    return {k: _apply_to_RUR(v) for k, v in d.items()}
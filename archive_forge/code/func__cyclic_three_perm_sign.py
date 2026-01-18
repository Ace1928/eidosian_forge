from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
@staticmethod
def _cyclic_three_perm_sign(v0, v1, v2):
    """
        Returns +1 or -1. It is +1 if and only if (v0, v1, v2) is in the
        orbit of (0, 1, 2) under the A4-action.
        """
    for t in [(v0, v1, v2), (v1, v2, v0), (v2, v0, v1)]:
        if t in [(0, 1, 2), (1, 3, 2), (2, 3, 0), (3, 1, 0)]:
            return +1
    return -1
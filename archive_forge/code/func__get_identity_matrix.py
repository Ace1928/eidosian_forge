from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _get_identity_matrix(self):
    N = self.N()
    return [[_kronecker_delta(i, j) for i in range(N)] for j in range(N)]
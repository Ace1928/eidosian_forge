from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _testing_assert_identity(self, m, allow_sign_if_obstruction_class=False):
    N = self.N()
    null = [[0 for i in range(N)] for j in range(N)]
    identity = self._get_identity_matrix()
    if allow_sign_if_obstruction_class and self.has_obstruction():
        if not (matrix.matrix_add(m, identity) == null or matrix.matrix_sub(m, identity) == null):
            raise Exception('Cocycle condition violated: %s' % m)
    elif not matrix.matrix_sub(m, identity) == null:
        raise Exception('Cocycle condition violated: %s' % m)
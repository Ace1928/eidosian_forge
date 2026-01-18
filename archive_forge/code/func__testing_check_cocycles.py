from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _testing_check_cocycles(self):
    for tet in range(self.num_tetrahedra()):
        for v in [(0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 3), (0, 3, 1), (0, 3, 2), (1, 0, 2), (1, 0, 3), (1, 2, 0), (1, 2, 3), (1, 3, 0), (1, 3, 2), (2, 0, 1), (2, 0, 3), (2, 1, 0), (2, 1, 3), (2, 3, 0), (2, 3, 1), (3, 0, 1), (3, 0, 2), (3, 1, 0), (3, 1, 2), (3, 2, 0), (3, 2, 1)]:
            m1 = self.middle_edge(tet, v[0], v[1], v[2])
            m2 = self.middle_edge(tet, v[0], v[2], v[1])
            self._testing_assert_identity(matrix.matrix_mult(m1, m2))
        for v in [(0, 1, 2), (0, 2, 1), (0, 3, 1), (1, 0, 2), (1, 2, 0), (1, 3, 0), (2, 0, 1), (2, 1, 0), (2, 3, 0), (3, 0, 1), (3, 1, 0), (3, 2, 0)]:
            m1 = self.long_edge(tet, v[0], v[1], v[2])
            m2 = self.long_edge(tet, v[1], v[0], v[2])
            self._testing_assert_identity(matrix.matrix_mult(m1, m2))
        for v in [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]:
            m1 = self.middle_edge(tet, v[0], v[1], v[2])
            m2 = self.middle_edge(tet, v[0], v[2], v[3])
            m3 = self.middle_edge(tet, v[0], v[3], v[1])
            self._testing_assert_identity(matrix.matrix_mult(m1, matrix.matrix_mult(m2, m3)))
        for v in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
            m1 = self.middle_edge(tet, v[0], v[1], v[2])
            m2 = self.long_edge(tet, v[0], v[2], v[1])
            m3 = self.middle_edge(tet, v[2], v[0], v[1])
            m4 = self.long_edge(tet, v[2], v[1], v[0])
            m5 = self.middle_edge(tet, v[1], v[2], v[0])
            m6 = self.long_edge(tet, v[1], v[0], v[2])
            self._testing_assert_identity(matrix.matrix_mult(m1, matrix.matrix_mult(m2, matrix.matrix_mult(m3, matrix.matrix_mult(m4, matrix.matrix_mult(m5, m6))))), True)
from ..matrix import vector, matrix
from ..math_basics import is_RealIntervalFieldElement
from ..sage_helper import _within_sage
from a real type (either a SnapPy.Number or one
def _det_shifted_matrix3(m, i):
    """
    Computes determinant of 3x3 matrix obtained by picking
    3 rows from the given 3x4 matrix m.
    """
    i0 = (i + 1) % 4
    i1 = (i + 2) % 4
    i2 = (i + 3) % 4
    return m[0][i0] * m[1][i1] * m[2][i2] + m[0][i1] * m[1][i2] * m[2][i0] + m[0][i2] * m[1][i0] * m[2][i1] - m[0][i2] * m[1][i1] * m[2][i0] - m[0][i0] * m[1][i2] * m[2][i1] - m[0][i1] * m[1][i0] * m[2][i2]
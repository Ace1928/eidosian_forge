from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
def _transform_points_to_make_first_one_infinity_and_inv_sl_matrix(idealPoints):
    z = idealPoints[0]
    CIF = z.parent()
    gl_matrix = matrix(CIF, [[0, 1], [1, -z]])
    sl_matrix = CIF(sage.all.I) * gl_matrix
    inv_sl_matrix = _adjoint2(sl_matrix)
    return ([apply_Moebius(gl_matrix, u) for u in idealPoints[1:]], inv_sl_matrix)
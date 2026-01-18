from ..upper_halfspace import psl2c_to_o13 # type: ignore
from ..upper_halfspace.ideal_point import ideal_point_to_r13 # type: ignore
from ..matrix import matrix # type: ignore
from ..math_basics import is_RealIntervalFieldElement # type: ignore
def _r13_fixed_points_of_psl2c_matrix(m):
    """
    Unguarded version of r13_fixed_points_of_psl2c_matrix.
    """
    return [ideal_point_to_r13(z, z.real().parent()) for z in _complex_fixed_points_of_psl2c_matrix(m)]
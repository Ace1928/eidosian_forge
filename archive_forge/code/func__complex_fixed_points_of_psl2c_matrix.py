from ..upper_halfspace import psl2c_to_o13 # type: ignore
from ..upper_halfspace.ideal_point import ideal_point_to_r13 # type: ignore
from ..matrix import matrix # type: ignore
from ..math_basics import is_RealIntervalFieldElement # type: ignore
def _complex_fixed_points_of_psl2c_matrix(m):
    """
    Given a PSL(2,C)-matrix acting on the upper halfspace H^3, compute
    the two fixed points as complex numbers on the boundary of H^3.
    """
    a = m[1, 0]
    b = m[1, 1] - m[0, 0]
    c = -m[0, 1]
    d = (b * b - 4 * a * c).sqrt()
    return [(-b + s * d) / (2 * a) for s in [+1, -1]]
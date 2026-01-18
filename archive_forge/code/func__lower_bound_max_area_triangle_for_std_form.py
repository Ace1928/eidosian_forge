from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _lower_bound_max_area_triangle_for_std_form(z):
    """
        Imagine an ideal tetrahedron in the upper half space model with
        vertices at 0, 1, z, and infinity. Pick the lowest (horizontal)
        horosphere about infinity that intersects the tetrahedron in a
        triangle, i.e, just touches the face opposite to infinity.
        This method will return the hyperbolic area of that triangle.

        The result is the same for z, 1/(1-z), and 1 - 1/z.
        """
    if z.real() < 0:
        return 2 * z.imag() / abs(z - 1) ** 2
    if z.real() > 1:
        return 2 * z.imag() / abs(z) ** 2
    if abs(2 * z - 1) < 1:
        return 2 * z.imag()
    return 2 * z.imag() ** 3 / (abs(z) * abs(z - 1)) ** 2
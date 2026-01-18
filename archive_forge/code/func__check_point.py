from .vector import Vector, _check_vector
from .frame import _check_frame
from warnings import warn
def _check_point(self, other):
    if not isinstance(other, Point):
        raise TypeError('A Point must be supplied')
from .sage_helper import _within_sage
from . import number
from .math_basics import is_Interval
def _noalgebra(self, other):
    raise TypeError('To do matrix algebra, please install numpy or run SnapPy in Sage.')
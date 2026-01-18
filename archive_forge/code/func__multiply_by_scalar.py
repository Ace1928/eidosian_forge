from .sage_helper import _within_sage
from . import number
from .math_basics import is_Interval
def _multiply_by_scalar(self, other):
    return SimpleMatrix([[other * e for e in row] for row in self.data])
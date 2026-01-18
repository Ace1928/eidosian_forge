from sympy.core.function import Function, ArgumentIndexError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos
class MathieuBase(Function):
    """
    Abstract base class for Mathieu functions.

    This class is meant to reduce code duplication.

    """
    unbranched = True

    def _eval_conjugate(self):
        a, q, z = self.args
        return self.func(a.conjugate(), q.conjugate(), z.conjugate())